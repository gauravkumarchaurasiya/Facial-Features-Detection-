import cv2
from roboflow import Roboflow
import requests
import numpy as np
import json

# Initialize the Roboflow model
rf = Roboflow(api_key="tdJrNi1RDnIUFiRCWVC5")
project = rf.workspace().project("face-features-0chll")
model = project.version(1).model

# Function to perform prediction on a single frame
def predict_frame(frame):
    # Save frame to a temporary file
    cv2.imwrite("temp.jpg", frame)
    
    # Perform prediction using the Roboflow model
    result = model.predict("temp.jpg", confidence=40, overlap=30).json()
    
    # Draw bounding boxes on the frame
    for prediction in result['predictions']:
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))
        color = (0, 255, 0)  # Green color for bounding box
        thickness = 2
        
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        frame = cv2.putText(frame, prediction['class'], (start_point[0], start_point[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to capture frames and perform predictions
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Perform prediction on the frame
    frame = predict_frame(frame)
    
    # Display the resulting frame
    cv2.imshow('Webcam', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
