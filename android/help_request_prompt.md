# Help request: SHARP ExecuTorch Vulkan export / Android runtime

## Context
- **Project:** Furnit Android app. SHARP model (4-part split) exported to ExecuTorch .pte with **Vulkan** backend for Part1/Part2/Part3/Part4.
- **Goal:** Verify export is correct; app sometimes fails with BackendFailed (error 32), device-lost, or mem-pressure/OOM during Part1+2.
- **Export:** Vulkan FP16, chunked Part4 (4a_512, 4a_65, 4b), patch_batch=2 for Part1/Part2.

## Exact export command
```
cd android
python3 export_sharp_executorch_split4.py \
  --backend vulkan \
  --chunked-part4 \
  --dtype fp16 \
  --patch-batch-size 2 \
  --sharp-src third_party/ml-sharp/src \
  --weights sharp_litert_models/sharp_2572gikvuh.pt \
  --output-dir executorch_models
```

## What I need help with
- Confirm from the export log below: did export complete for all parts (Part1, Part2, Part3, Part4, Part4a chunk 512/65, Part4b)? Any errors or warnings?
- If export succeeded but the app fails on device (BackendFailed 32, VK_DEVICE_LOST, or OOM): is that likely an export/runtime mismatch or device/driver/memory?

## Export log
Full log file: `/Users/al/Documents/tries01/Furnit/android/export_log_vulkan_20260318_233313.txt`

Last 400 lines:
```
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_bmm_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.bmm.default](args = (%aten_squeeze_copy_dims_2, %aten_squeeze_copy_dims_1), kwargs = {}):
INFO:root:   arg 1 (aten_squeeze_copy_dims_1): (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_view_copy_default_153 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.view_copy.default](args = (%aten_linear_default_3, [1, 65, 3, 16, 64]), kwargs = {}):
INFO:root:   arg 0 (aten_linear_default_3): (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.permute_copy.default
INFO:root:    aten.linear.default
INFO:root:    aten.add.Tensor
INFO:root:    aten.bmm.default
INFO:root:    aten.gelu.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    aten.squeeze_copy.dims
INFO:root:    aten.full_like.default
INFO:root:    et_vk.prepack.default
INFO:root:    aten.where.self
INFO:root:    aten.clone.default
INFO:root:    aten.native_layer_norm.default
INFO:root:    aten.view_copy.default
INFO:root:    aten.mul.Tensor
INFO:root:    aten._softmax.default
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.view_copy.default
INFO:root:    aten.squeeze_copy.dims
INFO:root:    aten.bmm.default
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_where_self : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.where.self](args = (%aten_logical_not_default_21, %aten_full_like_default, %aten__softmax_default), kwargs = {}):
INFO:root:   arg 1 (aten_full_like_default): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED))
INFO:root:   arg 2 (aten__softmax_default): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_bmm_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.bmm.default](args = (%aten_squeeze_copy_dims_2, %aten_squeeze_copy_dims_1), kwargs = {}):
INFO:root:   arg 1 (aten_squeeze_copy_dims_1): (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_view_copy_default_169 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.view_copy.default](args = (%aten_linear_default_3, [1, 65, 3, 16, 64]), kwargs = {}):
INFO:root:   arg 0 (aten_linear_default_3): (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.permute_copy.default
INFO:root:    aten.linear.default
INFO:root:    aten.add.Tensor
INFO:root:    aten.bmm.default
INFO:root:    aten.gelu.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    aten.squeeze_copy.dims
INFO:root:    aten.full_like.default
INFO:root:    et_vk.prepack.default
INFO:root:    aten.where.self
INFO:root:    aten.clone.default
INFO:root:    aten.native_layer_norm.default
INFO:root:    aten.view_copy.default
INFO:root:    aten.mul.Tensor
INFO:root:    aten._softmax.default
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.view_copy.default
INFO:root:    aten.squeeze_copy.dims
INFO:root:    aten.bmm.default
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_where_self : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.where.self](args = (%aten_logical_not_default_23, %aten_full_like_default, %aten__softmax_default), kwargs = {}):
INFO:root:   arg 1 (aten_full_like_default): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED))
INFO:root:   arg 2 (aten__softmax_default): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_bmm_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.bmm.default](args = (%aten_squeeze_copy_dims_2, %aten_squeeze_copy_dims_1), kwargs = {}):
INFO:root:   arg 1 (aten_squeeze_copy_dims_1): (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED))
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.permute_copy.default
INFO:root:    aten.linear.default
INFO:root:    aten.add.Tensor
INFO:root:    aten.bmm.default
INFO:root:    aten.gelu.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    aten.squeeze_copy.dims
INFO:root:    aten.full_like.default
INFO:root:    et_vk.prepack.default
INFO:root:    aten.where.self
INFO:root:    aten.clone.default
INFO:root:    aten.native_layer_norm.default
INFO:root:    aten.view_copy.default
INFO:root:    aten.mul.Tensor
INFO:root:    aten._softmax.default
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1]) = aten.log.default(torch.float32: torch.Size([1]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([]) = aten::scalar_tensor(0.04045,  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.bool: torch.Size([1, 3, 2, 768, 768]) = aten.bitwise_not.default(torch.bool: torch.Size([1, 3, 2, 768, 768]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.bool: torch.Size([1, 3, 2, 768, 768]) = aten.le.Scalar(torch.float32: torch.Size([1, 3, 2, 768, 768]), 0.04045,  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.log.default(torch.float32: torch.Size([1, 3, 2, 768, 768]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.reciprocal.default(torch.float32: torch.Size([1, 1, 2, 768, 768]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.bool: torch.Size([1, 1, 2, 768, 768]) = aten.gt.Scalar(torch.float32: torch.Size([1, 1, 2, 768, 768]), 20,  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.log1p.default(torch.float32: torch.Size([1, 1, 2, 768, 768]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.log.default(torch.float32: torch.Size([1, 1, 2, 768, 768]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no valid representations for op torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), [1, 3, 1, 1, 1],  ...)], skipping torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), [1, 3, 1, 1, 1],  ...)
INFO:root:[Vulkan Partitioner] Due to [no valid representations for op torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), [1, 3, 1, 1, 1],  ...)], skipping torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), [1, 3, 1, 1, 1],  ...)
INFO:root:[Vulkan Partitioner] Due to [no valid representations for op torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), [1, 3, 1, 1, 1],  ...)], skipping torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), [1, 3, 1, 1, 1],  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1, 2, 1536, 1536]) = aten.reciprocal.default(torch.float32: torch.Size([1, 2, 1536, 1536]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no valid representations for op torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 3, 1, 768, 768]), [1, 1, 2, 1, 1],  ...)], skipping torch.float32: torch.Size([1, 3, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 3, 1, 768, 768]), [1, 1, 2, 1, 1],  ...)
INFO:root:[Vulkan Partitioner] Due to [no valid representations for op torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.full_like.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), 1,  ...)], skipping torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.full_like.default(torch.float32: torch.Size([1, 1, 2, 768, 768]), 1,  ...)
INFO:root:[Vulkan Partitioner] Due to [no valid representations for op torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 1, 768, 768]), [1, 1, 2, 1, 1],  ...)], skipping torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 1, 768, 768]), [1, 1, 2, 1, 1],  ...)
INFO:root:[Vulkan Partitioner] Due to [no valid representations for op torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 1, 768, 768]), [1, 1, 2, 1, 1],  ...)], skipping torch.float32: torch.Size([1, 1, 2, 768, 768]) = aten.repeat.default(torch.float32: torch.Size([1, 1, 1, 768, 768]), [1, 1, 2, 1, 1],  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1, 1, 1536, 1536]) = aten.reciprocal.default(torch.float32: torch.Size([1, 1, 1536, 1536]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1, 1, 1536, 1536]) = aten.reciprocal.default(torch.float32: torch.Size([1, 1, 1536, 1536]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1]) = aten.reciprocal.default(torch.float32: torch.Size([1]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping torch.float32: torch.Size([1]) = aten.reciprocal.default(torch.float32: torch.Size([1]),  ...)
INFO:root:[Vulkan Partitioner] Due to [no operator implementation], skipping [torch.float32: torch.Size([1]), torch.int64: torch.Size([1])] = aten.min.dim(torch.float32: torch.Size([1, 4718592]), -1,  ...)
INFO:root:Found 8 Vulkan subgraphs to be partitioned.
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_view_copy_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.view_copy.default](args = (%aten_div_tensor, [1, 4718592]), kwargs = {}):
INFO:root:   arg 0 (aten_div_tensor): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.permute_copy.default
INFO:root:    et_vk.conv_with_clamp.default
INFO:root:    aten.full.default
INFO:root:    aten.add.Tensor
INFO:root:    aten.clamp.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    aten.relu.default
INFO:root:    aten.div.Tensor
INFO:root:    aten.convolution.default
INFO:root:    aten.clone.default
INFO:root:    aten.view_copy.default
INFO:root:    aten.cat.default
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    et_vk.prepack.default
INFO:root:    aten.add.Tensor
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.unsqueeze_copy.default
INFO:root:    aten.clamp.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    et_vk.prepack.default
INFO:root:    aten.mul.Tensor
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_mul_tensor_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.mul.Tensor](args = (%aten_arange_start_step, %dim_order_ops__to_dim_order_copy_default), kwargs = {}):
INFO:root:   arg 1 (dim_order_ops__to_dim_order_copy_default): (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_mul_tensor_3 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.mul.Tensor](args = (%aten_arange_start_step_1, %dim_order_ops__to_dim_order_copy_default_2), kwargs = {}):
INFO:root:   arg 1 (dim_order_ops__to_dim_order_copy_default_2): (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_div_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.div.Tensor](args = (%aten_mul_tensor_2, %dim_order_ops__to_dim_order_copy_default_1), kwargs = {}):
INFO:root:   arg 1 (dim_order_ops__to_dim_order_copy_default_1): (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_div_tensor_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.div.Tensor](args = (%aten_mul_tensor_3, %dim_order_ops__to_dim_order_copy_default_3), kwargs = {}):
INFO:root:   arg 1 (dim_order_ops__to_dim_order_copy_default_3): (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_unsqueeze_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.unsqueeze_copy.default](args = (%getitem, 2), kwargs = {}):
INFO:root:   arg 0 (getitem): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_unsqueeze_copy_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.unsqueeze_copy.default](args = (%getitem_1, 2), kwargs = {}):
INFO:root:   arg 0 (getitem_1): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_expand_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.expand_copy.default](args = (%aten_view_copy_default, [768, 768]), kwargs = {}):
INFO:root:   arg 0 (aten_view_copy_default): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_expand_copy_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.expand_copy.default](args = (%aten_view_copy_default_1, [768, 768]), kwargs = {}):
INFO:root:   arg 0 (aten_view_copy_default_1): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.expand_copy.default
INFO:root:    aten.unsqueeze_copy.default
INFO:root:    aten.sub.Tensor
INFO:root:    aten.max_pool2d_with_indices.default
INFO:root:    aten.mul.Tensor
INFO:root:    aten.arange.start_step
INFO:root:    et_vk.prepack.default
INFO:root:    aten.div.Tensor
INFO:root:    aten.clone.default
INFO:root:    aten.view_copy.default
INFO:root:    aten.cat.default
INFO:root:    dim_order_ops._to_dim_order_copy.default
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_unsqueeze_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.unsqueeze_copy.default](args = (%aten_avg_pool2d_default, 2), kwargs = {}):
INFO:root:   arg 0 (aten_avg_pool2d_default): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.avg_pool2d.default
INFO:root:    aten.add.Tensor
INFO:root:    aten.unsqueeze_copy.default
INFO:root:    aten.sub.Tensor
INFO:root:    aten.clamp.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    et_vk.prepack.default
INFO:root:    aten.sigmoid.default
INFO:root:    aten.clone.default
INFO:root:    aten.div.Tensor
INFO:root:    aten.neg.default
INFO:root:    aten.view_copy.default
INFO:root:    aten.mul.Tensor
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:[Vulkan Delegate] Inserting transition(s) for %et_vk_conv_with_clamp_default_10 : [num_users=2] = call_function[target=executorch.exir.dialects.edge._ops.et_vk.conv_with_clamp.default](args = (%aten_sub_tensor_1, %p_feature_model_image_encoder_conv_weight, %p_feature_model_image_encoder_conv_bias, [2, 2], [0, 0], [1, 1], False, [0, 0], 1, 0.0, 1.7976931348623157e+308), kwargs = {}):
INFO:root:   arg 0 (aten_sub_tensor_1): (TensorRepr(TEXTURE_3D, TENSOR_WIDTH_PACKED)) -> (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_view_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.view_copy.default](args = (%aten_convolution_default_43, [1, 11, 2, 768, 768]), kwargs = {}):
INFO:root:   arg 0 (aten_convolution_default_43): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:[Vulkan Delegate] Inserting transition(s) for %aten_view_copy_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.view_copy.default](args = (%aten_convolution_default_44, [1, 3, 2, 768, 768]), kwargs = {}):
INFO:root:   arg 0 (aten_convolution_default_44): (TensorRepr(TEXTURE_3D, TENSOR_CHANNELS_PACKED)) -> (TensorRepr(BUFFER, TENSOR_WIDTH_PACKED))
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.native_group_norm.default
INFO:root:    et_vk.conv_with_clamp.default
INFO:root:    aten.add.Tensor
INFO:root:    aten.sub.Tensor
INFO:root:    aten.unsqueeze_copy.default
INFO:root:    aten.clamp.default
INFO:root:    aten.relu.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    et_vk.prepack.default
INFO:root:    aten.div.Tensor
INFO:root:    aten.convolution.default
INFO:root:    aten.clone.default
INFO:root:    aten.exp.default
INFO:root:    aten.view_copy.default
INFO:root:    aten.cat.default
INFO:root:    aten.mul.Tensor
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.add.Tensor
INFO:root:    aten.sub.Tensor
INFO:root:    aten.slice_copy.Tensor
INFO:root:    et_vk.prepack.default
INFO:root:    aten.where.self
INFO:root:    aten.div.Tensor
INFO:root:    aten.sigmoid.default
INFO:root:    aten.mul.Tensor
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
INFO:root:Operators included in this Vulkan partition: 
INFO:root:    aten.permute_copy.default
INFO:root:    aten.pow.Tensor_Scalar
INFO:root:    aten.add.Tensor
INFO:root:    aten.unsqueeze_copy.default
INFO:root:    aten.slice_copy.Tensor
INFO:root:    aten.div.Tensor
INFO:root:    et_vk.prepack.default
INFO:root:    aten.where.self
INFO:root:    aten.sigmoid.default
INFO:root:    aten.select_copy.int
INFO:root:    aten.view_copy.default
INFO:root:    aten.cat.default
INFO:root:    aten.mul.Tensor
/opt/miniconda3/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
============================================================
Export 4-Part Split SHARP to ExecuTorch .pte (FP16)
Same architecture as LiteRT split - Android runs same pipeline
Backend: Vulkan GPU (20-60s)
Vulkan FP16: avoids INT8 staging crashes; patch_batch=2
============================================================

Loading SHARP...
  Fusing Conv+BN layers...
  Part 1: 153M params
  Part 2: 151M params
  Part 3: 153M params
  Part 4: 199M params

Validating split pipeline...
  35 patches
  Split pipeline: 1,179,648 Gaussians

============================================================
Exporting Part 1: Single-Patch Encoder A (blocks 0-11) + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 25, VulkanBackend id: 25
  FP16 export: 16s
  Saved: sharp_split_part1_vulkan_fp16.pte (291 MB)

============================================================
Exporting Part 2: Single-Patch Encoder B (blocks 12-23) + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 25, VulkanBackend id: 25
  FP16 export: 16s
  Saved: sharp_split_part2_vulkan_fp16.pte (289 MB)

============================================================
Exporting Part 3: Image Encoder A (blocks 0-11) + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 25, VulkanBackend id: 25
  FP16 export: 17s
  Saved: sharp_split_part3_vulkan_fp16.pte (291 MB)

============================================================
Exporting Part 4: Image Encoder B + Full Decoder + Gaussians + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 32, VulkanBackend id: 32
  FP32 export: 25s
  Saved: sharp_split_part4.pte (755 MB)

============================================================
Exporting Part 1 batch=2 (patch encoder A, Vulkan FP16) + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 26, VulkanBackend id: 26
  FP16 export: 18s
  Saved: sharp_split_part1_b2_vulkan_fp16.pte (291 MB)

============================================================
Exporting Part 2 batch=2 (patch encoder B, Vulkan FP16) + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 25, VulkanBackend id: 25
  FP16 export: 18s
  Saved: sharp_split_part2_b2_vulkan_fp16.pte (289 MB)
  Part1/Part2 batch=2 Vulkan FP16 exported (use in C++ when useVulkan)

============================================================
Exporting Part 4a chunk (512 tokens): ViT blocks 12-23 + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 25, VulkanBackend id: 25
  FP32 export: 17s
  Saved: sharp_split_part4a_chunk_512_vulkan.pte (577 MB)

============================================================
Exporting Part 4a chunk (65 tokens): ViT blocks 12-23 + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 25, VulkanBackend id: 25
  FP32 export: 17s
  Saved: sharp_split_part4a_chunk_65_vulkan.pte (577 MB)

============================================================
Exporting Part 4b: From tokens (577) + decoder + Gaussians + Vulkan GPU
============================================================
  Vulkan: using default planning (VulkanPartitioner AOT handles memory)
  [Partition] Vulkan strings in .pte: 8, VulkanBackend id: 8
  FP32 export: 8s
  Saved: sharp_split_part4b_vulkan.pte (178 MB)
  Chunked Part 4 output shape OK (Gaussians: 1,179,648)

============================================================
Export complete in 210s
============================================================
  part1: 291 MB
  part2: 289 MB
  part3: 291 MB
  part4: 755 MB
  part1_b2: 291 MB
  part2_b2: 289 MB
  part4a_chunk_512: 577 MB
  part4a_chunk_65: 577 MB
  part4b: 178 MB
  Total: 3537 MB
  Gaussians: 1,179,648

Push to device:
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part1.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part1_b2_vulkan_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part1_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part1_part2_combined.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part1_vulkan_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part2.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part2_b2_vulkan_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part2_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part2_vulkan_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part3.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part3_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part3_vulkan_fp16.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part4.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part4a_chunk_512.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part4a_chunk_512_vulkan.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part4a_chunk_65.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part4a_chunk_65_vulkan.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part4b.pte /sdcard/Android/data/com.furnit.android/files/models/
  adb push /Users/al/Documents/tries01/Furnit/android/executorch_models/sharp_split_part4b_vulkan.pte /sdcard/Android/data/com.furnit.android/files/models/

==============================================
Export finished at 2026-03-18 18:06:48 UTC — exit code 0
Full log written to: /Users/al/Documents/tries01/Furnit/android/export_log_vulkan_20260318_233313.txt
==============================================
```

---
If the log above is truncated, attach the full file: `/Users/al/Documents/tries01/Furnit/android/export_log_vulkan_20260318_233313.txt`
