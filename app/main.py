# created by Jackson Kim on 2025-05

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import face_recognition
from PIL import Image
import io

from app.recognition import recognize_faces

app = FastAPI(title="Face Recognition API")


@app.get("/")
def root():
    return {
        "message": "Face Recognition API Running"
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = recognize_faces(frame)

    return JSONResponse(content={
        "faces": results
    })