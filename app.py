from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os


# Create FastAPI app
app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model
model = YOLO("best.pt")
print("Model classes:", model.names)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure directories exist
os.makedirs("static/faces", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Required PPE classes
REQUIRED_CLASSES = {"no_helmet", "no_gloves", "no_goggles", "no_boots"}

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_path = f"static/{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(image_path)

    for result in results:
        img = result.plot()
        cv2.imwrite(image_path, img)

    return {"message": "Detection Complete", "image_url": f"/static/{file.filename}"}

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)

        for result in results:
            detected_labels = set(model.names[int(cls)] for cls in result.boxes.cls.tolist())
            missing = REQUIRED_CLASSES & detected_labels

            if missing:
                print(f"Missing PPE: {missing}")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                print(f"Number of faces detected: {len(faces)}")

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    face_img = frame[y:y + h, x:x + w]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"static/faces/face_{timestamp}.jpg"
                    cv2.imwrite(filename, face_img)
                    print(f"Saved face image: {filename}")
                    break

            frame = result.plot()

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/faces", response_class=HTMLResponse)
def view_faces(request: Request):
    face_dir = "static/faces"
    face_files = sorted(
        [f for f in os.listdir(face_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: os.path.getmtime(os.path.join(face_dir, x)),
        reverse=True
    )
    
    face_images = []
    for filename in face_files:
        timestamp = datetime.fromtimestamp(os.path.getmtime(os.path.join(face_dir, filename)))
        face_images.append({
            "url": f"/static/faces/{filename}",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })

    return templates.TemplateResponse("faces.html", {
        "request": request,
        "face_images": face_images
    })
    
    return HTMLResponse(content=html)
