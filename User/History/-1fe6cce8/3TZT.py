import numpy as np
import cv2

# resolution is 960p (2560x960)
cap = cv2.VideoCapture(0)
cap.set(3, 2560)
cap.set(4, 960)



# Load two images
imgL = cv2.imread("tsukuba_l.png", 0)
imgR = cv2.imread("tsukuba_r.png", 0)

# Create a stereo matcher object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute disparity
disparity = stereo.compute(imgL, imgR)

# Display disparity
cv2.imshow("disparity", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
