import torch
import torch.nn as nn
import copy
from ultralytics import YOLO


def inspect_detect_head(model):
    """Find the detect head and understand its structure."""
    print("="*60)
    print("Inspecting model structure for detect head...")
    print("="*60)
    
    for name, module in model.model.named_modules():
        # Find class prediction layers
        if isinstance(module, nn.Conv2d) and module.out_channels > 2000:
            print(f"  Conv2d: {name} → out_channels={module.out_channels}, "
                  f"kernel={module.kernel_size}")
        if isinstance(module, nn.Linear) and module.out_features > 2000:
            print(f"  Linear: {name} → out_features={module.out_features}")
    
    # Find the Detect/Segment head module
    detect_head = None
    detect_name = None
    for name, module in model.model.named_modules():
        class_name = type(module).__name__
        if class_name in ('Detect', 'Segment', 'WorldDetect', 
                          'YOLOEDetect', 'YOLOESegment'):
            detect_head = module
            detect_name = name
            print(f"\n  Found detect head: {name} ({class_name})")
            print(f"  Submodules:")
            for sub_name, sub_mod in module.named_modules():
                if sub_name and not '.' in sub_name:
                    print(f"    {sub_name}: {type(sub_mod).__name__}")
            break
    
    return detect_head, detect_name


class SplitClassHead(nn.Module):
    """
    Wraps the original class prediction conv/linear and splits it
    into two halves. Does NOT concat — returns both halves separately.
    The caller is responsible for handling both outputs.
    """
    def __init__(self, original_layer):
        super().__init__()
        
        if isinstance(original_layer, nn.Conv2d):
            out_ch = original_layer.out_channels
            in_ch = original_layer.in_channels
            self.split_at = out_ch // 2
            
            self.layer_a = nn.Conv2d(
                in_ch, self.split_at,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=original_layer.bias is not None
            )
            self.layer_b = nn.Conv2d(
                in_ch, out_ch - self.split_at,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=original_layer.bias is not None
            )
            
            with torch.no_grad():
                self.layer_a.weight.copy_(original_layer.weight[:self.split_at])
                self.layer_b.weight.copy_(original_layer.weight[self.split_at:])
                if original_layer.bias is not None:
                    self.layer_a.bias.copy_(original_layer.bias[:self.split_at])
                    self.layer_b.bias.copy_(original_layer.bias[self.split_at:])
        
        elif isinstance(original_layer, nn.Linear):
            out_f = original_layer.out_features
            in_f = original_layer.in_features
            self.split_at = out_f // 2
            
            self.layer_a = nn.Linear(in_f, self.split_at,
                                      bias=original_layer.bias is not None)
            self.layer_b = nn.Linear(in_f, out_f - self.split_at,
                                      bias=original_layer.bias is not None)
            
            with torch.no_grad():
                self.layer_a.weight.copy_(original_layer.weight[:self.split_at])
                self.layer_b.weight.copy_(original_layer.weight[self.split_at:])
                if original_layer.bias is not None:
                    self.layer_a.bias.copy_(original_layer.bias[:self.split_at])
                    self.layer_b.bias.copy_(original_layer.bias[self.split_at:])
        
        self.total_out = out_ch if isinstance(original_layer, nn.Conv2d) else out_f
        print(f"    Split: {type(original_layer).__name__}({self.total_out}) → "
              f"{self.split_at} + {self.total_out - self.split_at}")
    
    def forward(self, x):
        # Return TUPLE — not concatenated
        return self.layer_a(x), self.layer_b(x)


def patch_detect_forward(detect_head):
    """
    Monkey-patch the detect head's forward() to handle split class
    predictions with delayed concat (after permute, not before).
    
    This is model-architecture-specific. We need to understand how
    the original forward() works and insert the split logic.
    """
    original_forward = detect_head.forward
    
    # Store reference to the detect head for the closure
    head = detect_head
    
    def new_forward(x):
        """
        Modified forward that keeps class prediction halves separate
        through reshape/permute, only concatenating at the end.
        
        Original YOLO detect forward (simplified):
          for each scale i:
            box_pred = box_conv(x[i])         → [B, 4*reg, H, W]
            cls_pred = cls_conv(x[i])         → [B, nc, H, W]
            mask_pred = mask_conv(x[i])       → [B, 32, H, W]
            
            # Reshape each
            box = box_pred.view(B, 4*reg, -1)   → [B, 4*reg, H*W]
            cls = cls_pred.view(B, nc, -1)       → [B, nc, H*W]
            mask = mask_pred.view(B, 32, -1)     → [B, 32, H*W]
          
          # Concat across scales and features
          → [B, 4+nc+32, total_anchors]
        
        Modified version:
          cls_conv now returns (cls_a, cls_b) tuple
          We reshape each half separately:
            cls_a = cls_a.view(B, nc//2, -1)     → [B, 2293, H*W]
            cls_b = cls_b.view(B, nc-nc//2, -1)  → [B, 2292, H*W]
          Then concat cls_a + cls_b → [B, 4585, H*W]
          Then concat with box + mask as before
        """
        # Try to call original and intercept
        # This is fragile — better approach below
        return original_forward(x)
    
    # Instead of monkey-patching forward (fragile), we should
    # modify the class prediction path at the ONNX level
    # See patch_onnx_delayed_concat() below


def patch_onnx_delayed_concat(onnx_path: str, output_path: str):
    """
    More reliable approach: modify the ONNX graph directly.
    
    Find the pattern:
      conv_4585 → reshape → transpose
    Replace with:
      conv_2293 ─→ reshape_a → transpose_a ─┐
                                              ├→ concat
      conv_2292 ─→ reshape_b → transpose_b ─┘
    
    This keeps every tensor under 4096 channels through the transpose.
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import numpy as np
    
    print(f"Loading {onnx_path}...")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    initializers = {init.name: init for init in graph.initializer}
    
    # Build output→consumer map
    output_consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp not in output_consumers:
                output_consumers[inp] = []
            output_consumers[inp].append(node)
    
    # Build output→producer map
    output_producers = {}
    for node in graph.node:
        for out in node.output:
            output_producers[out] = node
    
    modifications = []
    
    for node in list(graph.node):
        if node.op_type not in ("Conv", "Gemm", "MatMul"):
            continue
        
        # Check if this is a 4585-channel op
        weight_name = node.input[1] if len(node.input) > 1 else None
        if weight_name not in initializers:
            continue
        
        weight = numpy_helper.to_array(initializers[weight_name])
        
        if node.op_type == "Conv":
            out_channels = weight.shape[0]
        elif node.op_type == "Gemm":
            # Gemm: Y = alpha * A * B + beta * C
            # Weight could be transposed
            transB = 0
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
            out_channels = weight.shape[0] if transB == 0 else weight.shape[1]
        else:
            continue
        
        if out_channels < 4096:
            continue
        
        print(f"\n  Found: {node.op_type} {node.name} out_channels={out_channels}")
        
        # Find what consumes this node's output
        conv_output = node.output[0]
        consumers = output_consumers.get(conv_output, [])
        
        print(f"  Consumers: {[f'{c.op_type}({c.name})' for c in consumers]}")
        
        # Find the Reshape/Transpose chain after this conv
        # Pattern: Conv → Reshape → Transpose (or Conv → Flatten → ...)
        reshape_node = None
        transpose_node = None
        
        for consumer in consumers:
            if consumer.op_type in ("Reshape", "Flatten"):
                reshape_node = consumer
                # Find transpose after reshape
                reshape_consumers = output_consumers.get(consumer.output[0], [])
                for rc in reshape_consumers:
                    if rc.op_type == "Transpose":
                        transpose_node = rc
                        break
            elif consumer.op_type == "Transpose":
                transpose_node = consumer
        
        if reshape_node or transpose_node:
            modifications.append({
                'conv_node': node,
                'weight_name': weight_name,
                'bias_name': node.input[2] if len(node.input) > 2 else None,
                'out_channels': out_channels,
                'reshape_node': reshape_node,
                'transpose_node': transpose_node,
                'conv_output': conv_output,
            })
            print(f"  → Will split conv and delay concat past "
                  f"{'reshape+transpose' if reshape_node and transpose_node else 'transpose'}")
        else:
            print(f"  → No reshape/transpose found downstream — using simple split+cat")
            modifications.append({
                'conv_node': node,
                'weight_name': weight_name,
                'bias_name': node.input[2] if len(node.input) > 2 else None,
                'out_channels': out_channels,
                'reshape_node': None,
                'transpose_node': None,
                'conv_output': conv_output,
            })
    
    print(f"\n{'='*60}")
    print(f"Applying {len(modifications)} modifications...")
    
    for mod in modifications:
        apply_delayed_concat_split(graph, mod, initializers, output_consumers)
    
    # Validate
    print("\nValidating ONNX graph...")
    try:
        onnx.checker.check_model(model)
        print("✅ Validation passed")
    except Exception as e:
        print(f"⚠️  Validation warning: {e}")
    
    onnx.save(model, output_path)
    print(f"✅ Saved: {output_path}")
    return True


def apply_delayed_concat_split(graph, mod, initializers, output_consumers):
    """
    Replace: Conv(4585) → Reshape → Transpose
    With:    Conv_a(2293) → Reshape_a → Transpose_a ─┐
             Conv_b(2292) → Reshape_b → Transpose_b ─├─ Concat
    """
    from onnx import helper, numpy_helper
    import numpy as np
    from copy import deepcopy
    
    conv = mod['conv_node']
    out_ch = mod['out_channels']
    split_at = out_ch // 2
    
    # Split weights
    orig_weight = numpy_helper.to_array(initializers[mod['weight_name']])
    weight_a = orig_weight[:split_at].copy()
    weight_b = orig_weight[split_at:].copy()
    
    wa_name = f"{mod['weight_name']}_a"
    wb_name = f"{mod['weight_name']}_b"
    graph.initializer.append(numpy_helper.from_array(weight_a, name=wa_name))
    graph.initializer.append(numpy_helper.from_array(weight_b, name=wb_name))
    
    # Split bias if present
    ba_name = ""
    bb_name = ""
    if mod['bias_name'] and mod['bias_name'] in initializers:
        orig_bias = numpy_helper.to_array(initializers[mod['bias_name']])
        graph.initializer.append(
            numpy_helper.from_array(orig_bias[:split_at].copy(), name=f"{mod['bias_name']}_a"))
        graph.initializer.append(
            numpy_helper.from_array(orig_bias[split_at:].copy(), name=f"{mod['bias_name']}_b"))
        ba_name = f"{mod['bias_name']}_a"
        bb_name = f"{mod['bias_name']}_b"
    
    conv_output = mod['conv_output']
    
    if mod['reshape_node'] and mod['transpose_node']:
        # DELAYED CONCAT: split through reshape+transpose
        reshape = mod['reshape_node']
        transpose = mod['transpose_node']
        final_output = transpose.output[0]  # this is what downstream ops consume
        
        # Create split conv outputs
        conv_a_out = f"{conv_output}_split_a"
        conv_b_out = f"{conv_output}_split_b"
        
        # Conv A
        conv_a_inputs = [conv.input[0], wa_name]
        if ba_name: conv_a_inputs.append(ba_name)
        conv_a = helper.make_node(conv.op_type, conv_a_inputs, [conv_a_out],
                                   name=f"{conv.name}_a")
        for attr in conv.attribute:
            conv_a.attribute.append(deepcopy(attr))
        
        # Conv B
        conv_b_inputs = [conv.input[0], wb_name]
        if bb_name: conv_b_inputs.append(bb_name)
        conv_b = helper.make_node(conv.op_type, conv_b_inputs, [conv_b_out],
                                   name=f"{conv.name}_b")
        for attr in conv.attribute:
            conv_b.attribute.append(deepcopy(attr))
        
        # Reshape A (need to adjust shape for split channels)
        reshape_a_out = f"{reshape.output[0]}_split_a"
        reshape_a_shape = f"{reshape.input[1]}_split_a" if len(reshape.input) > 1 else None
        
        # For reshape, we need to create new shape tensors
        # The original reshape goes from [B, 4585, H, W] to [B, 4585, H*W] or similar
        # We need [B, 2293, H*W] and [B, 2292, H*W]
        # Since the shape might be dynamic, we'll use Reshape with -1
        
        reshape_a = helper.make_node("Reshape", 
                                      [conv_a_out, reshape.input[1]] if len(reshape.input) > 1 
                                      else [conv_a_out],
                                      [reshape_a_out], name=f"{reshape.name}_a")
        for attr in reshape.attribute:
            reshape_a.attribute.append(deepcopy(attr))
        
        reshape_b_out = f"{reshape.output[0]}_split_b"
        reshape_b = helper.make_node("Reshape",
                                      [conv_b_out, reshape.input[1]] if len(reshape.input) > 1
                                      else [conv_b_out],
                                      [reshape_b_out], name=f"{reshape.name}_b")
        for attr in reshape.attribute:
            reshape_b.attribute.append(deepcopy(attr))
        
        # Transpose A and B
        transpose_a_out = f"{transpose.output[0]}_split_a"
        transpose_a = helper.make_node("Transpose", [reshape_a_out], [transpose_a_out],
                                        name=f"{transpose.name}_a")
        for attr in transpose.attribute:
            transpose_a.attribute.append(deepcopy(attr))
        
        transpose_b_out = f"{transpose.output[0]}_split_b"
        transpose_b = helper.make_node("Transpose", [reshape_b_out], [transpose_b_out],
                                        name=f"{transpose.name}_b")
        for attr in transpose.attribute:
            transpose_b.attribute.append(deepcopy(attr))
        
        # Concat (after transpose — this is the key!)
        # Determine concat axis (the channel dim after transpose)
        concat_axis = -1  # last dim typically, or could be 1
        for attr in transpose.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
                # Find which axis the channel dim ended up at
                # Original: [B, C, H*W], perm might be [0, 2, 1] → [B, H*W, C]
                # Channel was at dim 1, after perm it's at perm.index(1)... 
                # Actually just concat on the last dim to merge the channel halves
                concat_axis = len(perm) - 1
        
        concat = helper.make_node("Concat",
                                   [transpose_a_out, transpose_b_out],
                                   [final_output],  # same name as original transpose output
                                   name=f"{conv.name}_delayed_concat",
                                   axis=concat_axis)
        
        # Remove original nodes
        idx = list(graph.node).index(conv)
        graph.node.remove(conv)
        graph.node.remove(reshape)
        graph.node.remove(transpose)
        
        # Insert new nodes in order
        for new_node in [conv_a, conv_b, reshape_a, reshape_b, 
                         transpose_a, transpose_b, concat]:
            graph.node.insert(idx, new_node)
            idx += 1
        
        print(f"  ✅ Delayed concat: {conv.name} ({out_ch}) → "
              f"split({split_at}+{out_ch-split_at}) → reshape → transpose → concat")
    
    else:
        # Simple split+cat (no reshape/transpose to delay past)
        conv_a_out = f"{conv_output}_split_a"
        conv_b_out = f"{conv_output}_split_b"
        
        conv_a_inputs = [conv.input[0], wa_name]
        if ba_name: conv_a_inputs.append(ba_name)
        conv_a = helper.make_node(conv.op_type, conv_a_inputs, [conv_a_out],
                                   name=f"{conv.name}_a")
        for attr in conv.attribute:
            conv_a.attribute.append(deepcopy(attr))
        
        conv_b_inputs = [conv.input[0], wb_name]
        if bb_name: conv_b_inputs.append(bb_name)
        conv_b = helper.make_node(conv.op_type, conv_b_inputs, [conv_b_out],
                                   name=f"{conv.name}_b")
        for attr in conv.attribute:
            conv_b.attribute.append(deepcopy(attr))
        
        concat = helper.make_node("Concat",
                                   [conv_a_out, conv_b_out],
                                   [conv_output],
                                   name=f"{conv.name}_cat",
                                   axis=1)
        
        idx = list(graph.node).index(conv)
        graph.node.remove(conv)
        graph.node.insert(idx, conv_a)
        graph.node.insert(idx + 1, conv_b)
        graph.node.insert(idx + 2, concat)
        
        print(f"  ✅ Simple split: {conv.name} ({out_ch}) → "
              f"split({split_at}+{out_ch-split_at}) → concat")
    
    # Clean up old initializers
    for init in list(graph.initializer):
        if init.name == mod['weight_name']:
            graph.initializer.remove(init)
        if mod['bias_name'] and init.name == mod['bias_name']:
            graph.initializer.remove(init)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", 
                        default="/Users/al/Documents/tries01/Furnit/android/yoloe-11l-seg-pf.onnx")
    parser.add_argument("--output-onnx", default="yoloe-split-delayed.onnx")
    parser.add_argument("--output-mlpackage", default="yoloe-11l-seg-pf-ane.mlpackage")
    args = parser.parse_args()
    
    # Step 1: Split ONNX with delayed concat
    success = patch_onnx_delayed_concat(args.onnx, args.output_onnx)
    
    if success:
        # Step 2: Verify with onnxruntime
        import onnxruntime as ort
        import numpy as np
        
        print("\nVerifying outputs match...")
        np.random.seed(42)
        test_input = np.random.randn(1, 3, 1280, 1280).astype(np.float32)
        
        sess_orig = ort.InferenceSession(args.onnx)
        sess_split = ort.InferenceSession(args.output_onnx)
        
        input_name = sess_orig.get_inputs()[0].name
        out_orig = sess_orig.run(None, {input_name: test_input})
        out_split = sess_split.run(None, {input_name: test_input})
        
        for i, (a, b) in enumerate(zip(out_orig, out_split)):
            diff = np.max(np.abs(a - b))
            print(f"  Output[{i}] shape={a.shape} max_diff={diff:.8f}")
        
        # Step 3: Convert to CoreML
        import coremltools as ct
        import onnx

        # Load the ONNX model object first
        onnx_model = onnx.load(args.output_onnx)

        model = ct.convert(
            onnx_model,                       # ← pass the model OBJECT, not the file path
            inputs=[ct.ImageType(name="image", shape=(1, 3, 1280, 1280),
                                scale=1/255.0, bias=[0,0,0], color_layout="RGB")],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
            convert_to="mlprogram",
        )
        model.save(args.output_mlpackage)
        print(f"✅ Saved: {args.output_mlpackage}")
        
        # Print output names for iOS
        spec = model.get_spec()
        print("\nOutput names (update YoloEDetectionParser.swift):")
        for out in spec.description.output:
            print(f"  {out.name}")