import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        results = model(frame)  # predict on an image
