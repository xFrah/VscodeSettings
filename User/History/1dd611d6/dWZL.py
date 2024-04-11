import time
import camera_utils
import cv2 as cv
import new_led_utils

exp = 12
leds = new_led_utils.LEDs(start_yellow_loading=False)
camera = camera_utils.Camera(leds, fast_mode=True, exp=exp)

while True:
    try:
        frame = camera.grab_background()
        cv.imshow("asd", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("a"):
            exp -= 1
            camera = camera_utils.Camera(leds, fast_mode=True, exp=exp)
            time.sleep(5)
        elif key == ord("d"):
            exp += 1
            camera = camera_utils.Camera(leds, fast_mode=True, exp=exp)
    except Exception as e:
        print(e)
        break