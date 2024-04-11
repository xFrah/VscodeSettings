import numpy as np
import cv2

# resolution is 960p (2560x960)
cap = cv2.VideoCapture(0)
cap.set(3, 2560)
cap.set(4, 960)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 480))
    frame = frame[:, 160:1120]

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
