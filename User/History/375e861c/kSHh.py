import cv2
# load first frame of Sacco.mp4 using cv2 and imshow it
cap = cv2.VideoCapture('Sacco.mp4')
image = cap.read()[1]
# resize image to 0.15
image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
width = int(image.shape[1])
height = int(image.shape[0])
# scale frame by 0.15
cv2.imshow('Sacco', image)
cv2.setMouseCallback('Sacco', lambda event, x, y, flags, param: print(x / width, y / height))

while True:
    # show it but enable click callbacks, that should return the x and y coordinates of the mouse click, from 0 to 1
    cv2.imshow('Sacco', image)

    # wait for a key to be pressed
    cv2.waitKey(1)