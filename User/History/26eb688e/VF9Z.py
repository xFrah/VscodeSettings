import cv2

cap = cv2.VideoCapture(0)
gray = False
while True:
    ret, frame = cap.read()
    start=time.time()
    end=time.time()
    print(end-start)
    if frame is None:
        continue

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        gray = not gray
