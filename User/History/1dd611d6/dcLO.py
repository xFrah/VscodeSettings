import camera_utils
import cv2 as cv
import new_led_utils

leds = new_led_utils.LEDs()
camera = camera_utils.Camera(leds, fast_mode=True)

while True:
    frame = camera.grab_background()
    cv.imshow("asd", frame)
    cv.waitKey(1) & 0xFF
