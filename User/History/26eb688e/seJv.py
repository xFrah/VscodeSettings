import datetime
import cv2

cap = cv2.VideoCapture(0)
gray = False
counter = 0
start = datetime.datetime.now()
while True:
    ret, frame = cap.read()
    #if counter == 1000:
    #    counter = 0
    #    start=datetime.datetime.now()

    if frame is None:
        continue

    counter += 1
    end=datetime.datetime.now()
    print(counter / ((((end - start).microseconds)+1) / 1000000.0))

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        gray = not gray
