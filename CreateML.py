from ultralytics import YOLO

# Load the model
model = YOLO("yoloe-11l-seg-pf.pt")

# Export to CoreML with specified parameters
model.export(
    format="coreml",
    imgsz=960,
    batch=15,
    dynamic=True,
    nms=False,
    agnostic_nms=True,
    simplify=False,
    optimize=False
)

print("Export complete!")
