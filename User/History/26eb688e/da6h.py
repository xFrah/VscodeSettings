import time
import cv2

cap = cv2.VideoCapture(0)
gray = False
counter = 0
start = time.time()
while True:
    ret, frame = cap.read()
    if counter == 30:
        end = time.time() + 0.0000000001
        print(counter / (end - start))
        counter = 0
        start = time.time()

    if frame is None:
        continue

    counter += 1

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        gray = not gray
