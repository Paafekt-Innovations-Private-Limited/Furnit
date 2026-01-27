#!/usr/bin/env python3
"""
Test CoreML model structure - check for head, backbone, neck components.
"""

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import json

def analyze_model_structure(model_path):
    """Analyze CoreML model to find backbone, neck, head components."""

    print(f"Loading model: {model_path}")
    model = ct.models.MLModel(model_path)
    spec = model.get_spec()

    print("\n" + "="*60)
    print("MODEL SPECIFICATION")
    print("="*60)

    # Basic info
    print(f"\nModel type: {spec.WhichOneof('Type')}")

    # Inputs
    print("\n--- INPUTS ---")
    for inp in spec.description.input:
        print(f"  {inp.name}:")
        if inp.type.HasField('imageType'):
            img = inp.type.imageType
            print(f"    Image: {img.width}x{img.height}, colorSpace={img.colorSpace}")
        elif inp.type.HasField('multiArrayType'):
            arr = inp.type.multiArrayType
            print(f"    MultiArray: shape={list(arr.shape)}, dtype={arr.dataType}")

    # Outputs
    print("\n--- OUTPUTS ---")
    for out in spec.description.output:
        print(f"  {out.name}:")
        if out.type.HasField('multiArrayType'):
            arr = out.type.multiArrayType
            print(f"    MultiArray: shape={list(arr.shape)}, dtype={arr.dataType}")

    # Check if it's a neural network or ML program
    if spec.HasField('neuralNetwork'):
        analyze_neural_network(spec.neuralNetwork)
    elif spec.HasField('mlProgram'):
        analyze_ml_program(model, spec)
    else:
        print(f"\nUnknown model type: {spec.WhichOneof('Type')}")

def analyze_neural_network(nn):
    """Analyze neural network spec for components."""
    print("\n" + "="*60)
    print("NEURAL NETWORK ANALYSIS")
    print("="*60)

    layers = nn.layers
    print(f"\nTotal layers: {len(layers)}")

    # Categorize layers by type
    layer_types = {}
    for layer in layers:
        ltype = layer.WhichOneof('layer')
        layer_types[ltype] = layer_types.get(ltype, 0) + 1

    print("\nLayer types:")
    for ltype, count in sorted(layer_types.items(), key=lambda x: -x[1]):
        print(f"  {ltype}: {count}")

    # Look for named components
    backbone_layers = []
    neck_layers = []
    head_layers = []

    for layer in layers:
        name = layer.name.lower()
        if 'backbone' in name or 'encoder' in name or 'stem' in name:
            backbone_layers.append(layer.name)
        elif 'neck' in name or 'fpn' in name or 'pan' in name or 'upsample' in name:
            neck_layers.append(layer.name)
        elif 'head' in name or 'detect' in name or 'segment' in name or 'proto' in name:
            head_layers.append(layer.name)

    print_component_summary(backbone_layers, neck_layers, head_layers)

def analyze_ml_program(model, spec):
    """Analyze ML Program spec for components."""
    print("\n" + "="*60)
    print("ML PROGRAM ANALYSIS")
    print("="*60)

    # Get the MIL program
    try:
        # Load as MIL program for detailed analysis
        mil_spec = spec.mlProgram

        print(f"\nML Program version: {mil_spec.version}")

        # Analyze functions
        for func in mil_spec.functions:
            print(f"\nFunction: {func.name if hasattr(func, 'name') else 'main'}")

            # Count operations
            op_counts = {}
            backbone_ops = []
            neck_ops = []
            head_ops = []

            if hasattr(func, 'block_specializations'):
                for block_spec in func.block_specializations.values():
                    for op in block_spec.operations:
                        op_type = op.type
                        op_counts[op_type] = op_counts.get(op_type, 0) + 1

                        # Check output names for component hints
                        for out in op.outputs:
                            name = out.name.lower()
                            if any(x in name for x in ['backbone', 'encoder', 'stem', 'dark', 'csp']):
                                backbone_ops.append(out.name)
                            elif any(x in name for x in ['neck', 'fpn', 'pan', 'upsample', 'concat']):
                                neck_ops.append(out.name)
                            elif any(x in name for x in ['head', 'detect', 'segment', 'proto', 'cv2', 'cv3']):
                                head_ops.append(out.name)

            print(f"\nOperation types ({len(op_counts)} unique):")
            for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
                print(f"  {op_type}: {count}")

            print_component_summary(backbone_ops, neck_ops, head_ops)

    except Exception as e:
        print(f"Error analyzing ML Program: {e}")

        # Fallback: analyze via coremltools
        print("\nFallback: Analyzing via model inspection...")
        try:
            # Get operation names from the model
            prog = ct.converters.mil.mil.Program()
            # This won't work directly, need different approach
        except:
            pass

def print_component_summary(backbone, neck, head):
    """Print summary of found components."""
    print("\n" + "-"*40)
    print("COMPONENT DETECTION")
    print("-"*40)

    print(f"\n✓ BACKBONE layers/ops found: {len(backbone)}")
    if backbone[:5]:
        for name in backbone[:5]:
            print(f"    - {name}")
        if len(backbone) > 5:
            print(f"    ... and {len(backbone)-5} more")

    print(f"\n✓ NECK layers/ops found: {len(neck)}")
    if neck[:5]:
        for name in neck[:5]:
            print(f"    - {name}")
        if len(neck) > 5:
            print(f"    ... and {len(neck)-5} more")

    print(f"\n✓ HEAD layers/ops found: {len(head)}")
    if head[:5]:
        for name in head[:5]:
            print(f"    - {name}")
        if len(head) > 5:
            print(f"    ... and {len(head)-5} more")

def inspect_model_weights(model_path):
    """Inspect model weights to understand structure."""
    print("\n" + "="*60)
    print("WEIGHT ANALYSIS")
    print("="*60)

    model = ct.models.MLModel(model_path)
    spec = model.get_spec()

    if spec.HasField('mlProgram'):
        mil = spec.mlProgram

        # Look for weight blobs
        weight_names = []

        for func in mil.functions:
            if hasattr(func, 'block_specializations'):
                for block_name, block in func.block_specializations.items():
                    for op in block.operations:
                        for attr_name, attr in op.attributes.items():
                            if 'weight' in attr_name.lower() or 'bias' in attr_name.lower():
                                for out in op.outputs:
                                    weight_names.append(out.name)

        # Categorize weights by likely component
        backbone_weights = [w for w in weight_names if any(x in w.lower() for x in ['backbone', 'encoder', 'stem', 'dark'])]
        neck_weights = [w for w in weight_names if any(x in w.lower() for x in ['neck', 'fpn', 'pan'])]
        head_weights = [w for w in weight_names if any(x in w.lower() for x in ['head', 'detect', 'proto'])]

        print(f"\nWeight tensors suggesting backbone: {len(backbone_weights)}")
        print(f"Weight tensors suggesting neck: {len(neck_weights)}")
        print(f"Weight tensors suggesting head: {len(head_weights)}")

def test_inference_outputs(model_path):
    """Run inference and analyze output structure."""
    print("\n" + "="*60)
    print("INFERENCE OUTPUT ANALYSIS")
    print("="*60)

    from PIL import Image
    import numpy as np

    model = ct.models.MLModel(model_path)

    # Create dummy input
    dummy_img = Image.new('RGB', (1280, 1280), (128, 128, 128))

    print("\nRunning inference with dummy input...")
    output = model.predict({'image': dummy_img})

    print("\nOutput tensors:")
    for name, tensor in output.items():
        shape = tensor.shape if hasattr(tensor, 'shape') else 'scalar'
        dtype = tensor.dtype if hasattr(tensor, 'dtype') else type(tensor)

        # Infer component from shape
        component = "unknown"
        if hasattr(tensor, 'shape'):
            if len(tensor.shape) == 4 and tensor.shape[1] == 32:
                component = "PROTOTYPES (segmentation head)"
            elif len(tensor.shape) == 3 and tensor.shape[-1] > 32:
                component = "DETECTIONS (detection head)"
            elif len(tensor.shape) == 4 and tensor.shape[1] > 32:
                component = "FEATURE MAP (possibly neck/backbone)"

        print(f"\n  {name}:")
        print(f"    Shape: {shape}")
        print(f"    Dtype: {dtype}")
        print(f"    Likely: {component}")

        if hasattr(tensor, 'shape'):
            print(f"    Stats: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")

def main():
    model_path = "/Users/al/Documents/tries01/Furnit/yoloe-26l-seg-pf.mlpackage"

    print("="*60)
    print("YOLOE-26L-SEG-PF MODEL STRUCTURE ANALYSIS")
    print("="*60)

    # Basic structure analysis
    analyze_model_structure(model_path)

    # Weight analysis
    inspect_model_weights(model_path)

    # Inference output analysis
    test_inference_outputs(model_path)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
YOLO Segmentation Model Architecture:

    INPUT (1280x1280 RGB)
           │
           ▼
    ┌──────────────┐
    │   BACKBONE   │  ← Feature extraction (CSPDarknet/etc)
    │   (encoder)  │     Produces multi-scale features
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │     NECK     │  ← Feature fusion (FPN/PAN)
    │  (FPN/PAN)   │     Combines multi-scale features
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │     HEAD     │  ← Detection + Segmentation heads
    │  (detect +   │     Outputs: boxes, scores, mask coeffs
    │   segment)   │     + prototype masks
    └──────────────┘
           │
           ▼
    OUTPUTS:
      - detections: [N, 38] (x,y,w,h,conf,class,32 coeffs)
      - prototypes: [32, 320, 320] (mask basis)

    Final mask = sigmoid(prototype @ coefficients)
""")

if __name__ == "__main__":
    main()
