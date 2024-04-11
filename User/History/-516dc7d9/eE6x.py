import cv2 as cv
import os
import pickle

import numpy as np

if __name__ == "__main__":
    # deserialize
    with open("camera0_data.pkl", "rb") as f:
        cmtx0, dist0, R0, T0 = pickle.load(f)
    with open("camera1_data.pkl", "rb") as f:
        cmtx1, dist1, R1, T1 = pickle.load(f)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # create disparity window and trackbars
    cv.namedWindow("disparity")
    cv.createTrackbar("numDisparities", "disparity", 16, 64, lambda x: None)
    cv.createTrackbar("blockSize", "disparity", 15, 21, lambda x: None)

    # callback for numDisparities
    def numDisparities_callback(x):
        global nd
        nd = x

    # callback for blockSize
    def blockSize_callback(x):
        global bs
        bs = x

    # set callbacks
    cv.setTrackbarPos("numDisparities", "disparity", 16)
    cv.setTrackbarPos("blockSize", "disparity", 15)

    nd = 16
    bs = 15

    while True:
        ret, frame = cap.read()

        # convert to grayscale
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # rotate 180
        frame = cv.rotate(frame, cv.ROTATE_180).copy()

        imgL = frame[:, :640]
        imgR = frame[:, 640:]

        # use undistort to remove distortion
        imgL_u = cv.undistort(imgL, cmtx0, dist0)
        imgR_u = cv.undistort(imgR, cmtx1, dist1)

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        min_disp = -128
        max_disp = 128
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(imgL_u, imgR_u)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)
        # show both images
        cv.imshow("left", imgL_u)
        cv.imshow("right", imgR_u)
        cv.imshow("disparity", disparity)
        cv.waitKey(1)
