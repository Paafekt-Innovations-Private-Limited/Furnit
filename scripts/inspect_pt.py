#!/usr/bin/env python3
import sys
import torch
from pathlib import Path

p = Path(sys.argv[1])
print('Inspecting', p)
try:
    obj = torch.jit.load(str(p), map_location='cpu')
    print('Loaded with torch.jit.load. Type:', type(obj))
    try:
        print('Has state_dict?', hasattr(obj, 'state_dict'))
    except Exception:
        pass
except Exception as e:
    print('torch.jit.load failed:', repr(e))
    try:
        obj = torch.load(str(p), map_location='cpu')
        print('Loaded with torch.load. Type:', type(obj))
        if isinstance(obj, dict):
            print('Dictionary keys (first 50):')
            ks = list(obj.keys())
            for k in ks[:50]:
                print('  -', k)
            print('Total keys:', len(ks))
        else:
            print('Object repr (truncated):')
            s = repr(obj)
            print(s[:1000])
    except Exception as e2:
        print('torch.load also failed:', repr(e2))
        raise SystemExit(2)

print('Done')
