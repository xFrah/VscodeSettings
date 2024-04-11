import camera_utils
import cv2 as cv
import new_led_utils

leds = new_led_utils.LEDs()
camera = camera_utils.Camera(leds, fast_mode=True, exp=12)

while True:
    frame = camera.grab_background()
    cv.imshow("asd", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("a"):
        camera = camera_utils.Camera(leds, fast_mode=True, exp=11)

