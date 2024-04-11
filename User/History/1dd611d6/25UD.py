import time
import camera_utils
import cv2 as cv
import new_led_utils

exp = 12
leds = new_led_utils.LEDs(start_yellow_loading=False)
cap = cv.VideoCapture(0)

succ[cv.CAP_PROP_FRAME_WIDTH] = self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
succ[cv.CAP_PROP_FRAME_HEIGHT] = self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
succ[cv.CAP_PROP_FPS] = self.cap.set(cv.CAP_PROP_FPS, 120)
time.sleep(2)
succ[cv.CAP_PROP_FOURCC] = self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# time.sleep(2)
succ[cv.CAP_PROP_AUTO_EXPOSURE] = self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
time.sleep(2)
succ[cv.CAP_PROP_AUTO_WB] = self.cap.set(cv.CAP_PROP_AUTO_WB, 0)
succ[cv.CAP_PROP_EXPOSURE] = self.cap.set(cv.CAP_PROP_EXPOSURE, self.exp)
succ[cv.CAP_PROP_GAIN] = self.cap.set(cv.CAP_PROP_GAIN, 100)

