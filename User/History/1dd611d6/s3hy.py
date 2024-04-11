import time
import camera_utils
import cv2 as cv
import new_led_utils

exp = 12
leds = new_led_utils.LEDs(start_yellow_loading=False)
cap = cv.VideoCapture(0)

succ = dict()
