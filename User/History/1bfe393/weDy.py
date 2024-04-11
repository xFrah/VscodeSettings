from lib_ws2812 import WS2812
import time
import _thread


class LEDs:
    def __init__(self, pin, count, intensity=1.0):
        self.ws2812 = WS2812(pin, count)
        self.animation
        self.intensity = intensity

    def thread(self, data):
        while True:
            try:
                if len(self.animations) > 0:
                    self.animations.pop(0)()
            except Exception as e:
                print("[LED] Error while displaying data", e)
            time.sleep(0.1)

    def animation(self, animation_type):

