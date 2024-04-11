import cv2
# load first frame of Sacco.mp4 using cv2 and imshow it
cap = cv2.VideoCapture('Sacco.mp4')
cv2.setMouseCallback('Sacco', lambda event, x, y, flags, param: print(x, y))

while True:
    # show it but enable click callbacks, that should return the x and y coordinates of the mouse click, from 0 to 1
    cv2.imshow('Sacco', cap.read()[1])

    # wait for a key to be pressed
    cv2.waitKey(1)