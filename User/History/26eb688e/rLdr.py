import datetime
import cv2

cap = cv2.VideoCapture(0)
gray = False
counter = 0
start = datetime.datetime.now()
while True:
    ret, frame = cap.read()
    if counter == 10:
        counter = 0

    if frame is None:
        continue

    counter += 1
    end = time.time()
    print(counter / (end - start))

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        gray = not gray
