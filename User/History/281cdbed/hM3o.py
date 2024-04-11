import cv2
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

cap = cv2.VideoCapture(0)


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{class_id} ({confidence:.2f})"
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


while True:
    ret, frame = cap.read()
    if ret:
        outputs = model(frame)  # predict on an image

        