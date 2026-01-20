#!/usr/bin/env python3
import sys
from pathlib import Path

p = Path(sys.argv[1])
print('Loading with safe globals:', p)
import torch
import sys
from pathlib import Path
# Insert local stub package into path so pickle can import ultralytics.nn.tasks
# Ensure local folder is on path but after system packages so installed `ultralytics` takes precedence
sys.path.append(str(Path(__file__).parent))
try:
    from ultralytics.nn.tasks import YOLOESegModel
    torch.serialization.add_safe_globals([YOLOESegModel])
    print('Registered YOLOESegModel as safe global (local stub)')
except Exception as e:
    print('Could not import YOLOESegModel:', e)

try:
    obj = torch.load(str(p), map_location='cpu', weights_only=False)
    print('torch.load succeeded. Type:', type(obj))
    try:
        print('Has state_dict:', hasattr(obj, 'state_dict'))
        if hasattr(obj, 'state_dict'):
            sd = obj.state_dict()
            print('state_dict keys (first 50):', list(sd.keys())[:50])
    except Exception as e:
        print('state_dict inspect failed:', e)
except Exception as e:
    print('torch.load failed:', repr(e))
    raise SystemExit(2)

print('Done')
