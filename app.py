from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()  

# Roboflow Setup

app = FastAPI()

# Serve static files from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html manually on root
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace().project("face-features-0chll")
model = project.version(1).model

# Frame processing function
def process_frame(base64_image):
    nparr = np.frombuffer(base64.b64decode(base64_image), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("temp.jpg", frame)

    result = model.predict("temp.jpg", confidence=40, overlap=30).json()

    for prediction in result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        start_point = (int(x - w / 2), int(y - h / 2))
        end_point = (int(x + w / 2), int(y + h / 2))

        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(frame, prediction["class"], (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

# WebSocket for real-time video
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        processed_img = process_frame(data)
        await websocket.send_text(processed_img)
