from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yoloe-26l-seg-pf.pt")

# Export directly without manual model.fuse()
success_path = model.export(
    format="coreml",
    imgsz=1280,
    batch=15,
    dynamic=True,
    nms=False,      # YOLO26 is natively end-to-end
    half=False,     # Maintains FP32 precision for textures
    simplify=True
)

print(f"Export complete: {success_path}")
