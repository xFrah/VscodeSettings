import cv2
# load first frame of Sacco.mp4 using cv2 and imshow it
cap = cv2.VideoCapture('Sacco.mp4')

ret, frame = cap.read()
cv2.imshow('frame', frame)
