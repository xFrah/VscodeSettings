import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from TOF import Tof

cap = cv2.VideoCapture(0)

print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280))
print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480))
print(cap.set(cv2.CAP_PROP_FPS, 30))
time.sleep(2)
# print image prop width and height
print(cap.get(3))
print(cap.get(4))

mode = 0

# Global variables to store the points and image
src0 = np.float32([(25, 25), (75, 25), (75, 75), (25, 75)])
src1 = np.float32([(25, 25), (75, 25), (75, 75), (25, 75)])
pts0 = src0.copy()
pts1 = src0.copy()
current_moving_point_index = -1
dst0 = np.float32(pts0)
dst1 = np.float32(pts1)

tof = Tof("/dev/ttyUSB1", background_shape=(480, 1280))
tof2 = Tof("/dev/ttyUSB3", background_shape=(480, 1280))


# Callback function for the mouse events
def mouse_callback(event, x, y, flags, param):
    global pts0, pts1, current_moving_point_index

    if mode == 0:
        src = src0
        pts = pts0
    else:
        src = src1
        pts = pts1

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if np.sqrt((x - pts[i][0]) ** 2 + (y - pts[i][1]) ** 2) < 10:
                current_moving_point_index = i
                print("Moving point {}".format(i))
    elif event == cv2.EVENT_LBUTTONUP:
        current_moving_point_index = -1
    elif event == cv2.EVENT_MOUSEMOVE and current_moving_point_index != -1:
        pts[current_moving_point_index] = (x, y)
        dst = np.float32(pts)
        selected_tof.change_warp_matrix(src, dst)


# Create a window and set the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

image = None
selected_tof = tof
while True:

    if mode == 0:
        src = src0
        pts = pts0
    else:
        src = src1
        pts = pts1

    image_t = selected_tof.get_image()
    if image_t is not None:
        image = selected_tof.color_image(image_t)

    if image is None:
        continue

    frame_t = cap.read()[1]
    if frame_t is not None:
        frame = frame_t
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    warped = selected_tof.warp_image(image)
    warped = cv2.addWeighted(frame, 1, warped, 1, 0)

    for i in range(4):
        cv2.circle(warped, (int(pts[i][0]), int(pts[i][1])), 10, (0, 0, 255, 255), 1)

    cv2.imshow("Image", warped)

    key = cv2.waitKey(1)
    if key == ord('q'):
        selected_tof = tof
        mode = 0
        print("Mode 0")
    elif key == ord('w'):
        selected_tof = tof2
        mode = 1
        print("Mode 1")
    elif key == ord('s'):
        name = selected_tof.addr + "-M.npy"
        np.save(name, selected_tof.M)
        print(f"Saved {name} to file")
    elif key == ord('d'):
        name = selected_tof.addr + "-M.npy"
        M0 = np.load(name)
        print(f"Loaded {name} from file")
