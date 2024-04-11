import camera_utils
import cv2 as cv
import new_led_utils

exp = 12
leds = new_led_utils.LEDs()
camera = camera_utils.Camera(leds, fast_mode=True, exp=exp)

while True:
    frame = camera.grab_background()
    cv.imshow("asd", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("a"):
        exp -= 1
        camera = camera_utils.Camera(leds, fast_mode=True, exp=exp)
    elif key == ord("d"):
        exp += 1
        camera = camera_utils.Camera(leds, fast_mode=True, exp=exp)

