import cv2 as cv
import os
import pickle

if __name__ == "__main__":
    # deserialize
    with open("camera0_data.pkl", "rb") as f:
        cmtx0, dist0, R0, T0 = pickle.load(f)
    with open("camera1_data.pkl", "rb") as f:
        cmtx1, dist1, R1, T1 = pickle.load(f)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # callback for numDisparities
    def numDisparities_callback(x):
        global nd
        nd = x

    # callback for blockSize
    def blockSize_callback(x):
        global bs
        bs = x

    nd = 16
    bs = 15

    # create disparity window and trackbars
    cv.namedWindow("disparity")
    cv.createTrackbar("numDisparities", "disparity", nd, 64, numDisparities_callback)
    cv.createTrackbar("blockSize", "disparity", bs, 21, blockSize_callback)

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

        # compute disparity
        stereo = cv.StereoBM_create(numDisparities=nd, blockSize=bs)
        disparity = stereo.compute(imgL_u, imgR_u)

        # normalize disparity
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        # show both images
        cv.imshow("left", imgL_u)
        cv.imshow("right", imgR_u)
        cv.imshow("disparity", disparity)
        cv.waitKey(1)
