import datetime
import time
import camera_utils
import cv2 as cv
import new_led_utils

exp = 12
leds = new_led_utils.LEDs(start_yellow_loading=False)
leds

start = datetime.datetime.now()
cap = cv.VideoCapture(0)

succ = dict()

succ[cv.CAP_PROP_FRAME_WIDTH] = cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
succ[cv.CAP_PROP_FRAME_HEIGHT] = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
succ[cv.CAP_PROP_FPS] = cap.set(cv.CAP_PROP_FPS, 120)
time.sleep(2)
succ[cv.CAP_PROP_FOURCC] = cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# time.sleep(2)
succ[cv.CAP_PROP_AUTO_EXPOSURE] = cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
time.sleep(2)
succ[cv.CAP_PROP_AUTO_WB] = cap.set(cv.CAP_PROP_AUTO_WB, 0)
succ[cv.CAP_PROP_EXPOSURE] = cap.set(cv.CAP_PROP_EXPOSURE, exp)
succ[cv.CAP_PROP_GAIN] = cap.set(cv.CAP_PROP_GAIN, 100)

print(f"Done, {str(tuple([round(100 / (datetime.datetime.now() - start).total_seconds(), 2)] + [round(cap.get(item), 2) if value else 'FAILED' for item, value in succ.items()]))}")
leds.change_to_white()

while True:
    _, frame = cap.read()
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord('a'):
        # change camera exposure
        exp += 1
        succ[cv.CAP_PROP_EXPOSURE] = cap.set(cv.CAP_PROP_EXPOSURE, exp)
    elif cv.waitKey(1) == ord('s'):
        exp -= 1
        print("Exposure set to ", cap.set(cv.CAP_PROP_EXPOSURE, exp))
