from ultralytics import YOLO
from ultralytics.nn.modules import head

# 1. Patch fuse methods to be no-ops BEFORE loading
for name in dir(head):
    cls = getattr(head, name)
    if isinstance(cls, type) and hasattr(cls, 'fuse'):
        cls.fuse = lambda self, *args, **kwargs: None
        print(f"Disabled {name}.fuse()")

# 2. Load model
model = YOLO("yoloe-26l-seg-pf.pt")

# 3. Export to CoreML
success_path = model.export(
    format="coreml",
    imgsz=1280,
    batch=1,
    nms=False,
    half=False,
    simplify=True
)

print(f"Done! Exported to: {success_path}")
