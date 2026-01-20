#!/usr/bin/env python3
import sys
from pathlib import Path

pt = Path(sys.argv[1])
onnx_out = Path(sys.argv[2])
dummy_shape = tuple(int(x) for x in (sys.argv[3].split(',') if len(sys.argv)>3 else '1,3,1536,1536'.split(',')))

print('Loading checkpoint', pt)
from ultralytics.nn import load_checkpoint
ck = load_checkpoint(str(pt))
model_obj = ck[0]
meta = ck[1]
print('Loaded types:', type(model_obj), type(meta))

# ultralytics model may store actual nn.Module at model_obj.model
try:
    module = model_obj.model
    print('Using module = model_obj.model')
except Exception:
    module = model_obj
    print('Using module = model_obj')

import torch
module.eval()
module.to('cpu')

dummy = torch.zeros(dummy_shape)
print('Exporting ONNX to', onnx_out)
try:
    torch.onnx.export(module, dummy, str(onnx_out), opset_version=13, input_names=['input'], output_names=['output'], dynamic_axes=None)
    print('ONNX export success')
except Exception as e:
    print('ONNX export failed:', e)
    raise
