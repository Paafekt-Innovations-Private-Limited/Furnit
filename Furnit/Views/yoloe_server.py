"""
YOLOE REST API server using FastAPI and Ultralytics
Save this as yoloe_server.py and run with: uvicorn yoloe_server:app --reload
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
from typing import List

app = FastAPI()

# Load your YOLOE or YOLOv8 model – replace with your weights as needed
model = YOLO('yoloe-11l-seg-pf.pt')  # Large YOLOE; adjust path if needed

@app.post('/detect')
async def detect(file: UploadFile = File(...)):
    # Read and process uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Run inference
    results = model(image)
    
    # Convert detections to JSON
    response = results[0].tojson()
    return JSONResponse(content=response)

@app.post('/detect_details')
async def detect_details(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    results = model(image)
    boxes_list = []
    for r in results:
        boxes = r.boxes.xyxyn.cpu().tolist() if hasattr(r.boxes, 'xyxyn') else r.boxes.xyxy.cpu().tolist()
        cls = r.boxes.cls.cpu().tolist()
        for b, c in zip(boxes, cls):
            boxes_list.append({'box': b, 'class': int(c), 'class_name': model.names[int(c)]})
    return {'results': boxes_list}

# To run:
# pip install fastapi uvicorn ultralytics pillow
# uvicorn yoloe_server:app --reload
#
# Then POST images to http://localhost:8000/detect

