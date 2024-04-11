import cv2

cap = cv2.VideoCapture(0)
gray = False
counter=0
while True:
    ret, frame = cap.read()
    counter+=1
    print(counter)
    
    

    print(end-start)
    if frame is None:
        counter-=1
        continue

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        gray = not gray
