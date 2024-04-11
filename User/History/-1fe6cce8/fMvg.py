import numpy as np
import cv2

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

# porco dio che schifo di codice che ho scritto ma funziona quindi non me ne frega niente, non lo voglio più vedere questo codice di merda che ho scritto
# ma perché cazzo non funziona con le immagini che ho fatto io? ma che ne so.
# comunque, secondo me il problema è che le immagini che ho fatto io non sono in scala di grigi, ma non so come fare a convertirle in scala di grigi
# quindi ho preso due immagini a caso da internet e le ho usate per fare il codice, e funziona, quindi non me ne frega niente, non lo voglio più vedere
# porca