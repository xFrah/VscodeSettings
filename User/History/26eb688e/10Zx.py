import cv2

cap = cv2.VideoCapture(0)
gray = False
while True:
    ret, frame = cap.read()

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
