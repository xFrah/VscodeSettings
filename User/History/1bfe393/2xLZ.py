from lib_ws2812 import WS2812
import time
import _thread


class LEDs:
    def __init__(self, pin, count, intensity=1.0):
        self.ws2812 = WS2812(pin, count)
        self.animation = None
        self.intensity = intensity

    def thread(self, data):
        while True:
            try:
                while self.animation == "pulsing_red":
                    for i in range(0, 255):
                        self.ws2812.show([(i, 0, 0)] * self.ws2812.led_count)
                        time.sleep(0.01)
                    for i in range(255, 0, -1):
                        self.ws2812.show([(i, 0, 0)] * self.ws2812.led_count)
                        time.sleep(0.01)
            except Exception as e:
                print("[LED] Error while displaying data", e)
            time.sleep(0.1)

    def animation(self, animation_type: str):
        def pulsing_red():
            while self.animation == "pulsing_red":
                for i in range(0, 255):
                    self.ws2812.show([(i, 0, 0)] * self.ws2812.led_count)
                    time.sleep(0.01)
                for i in range(255, 0, -1):
                    self.ws2812.show([(i, 0, 0)] * self.ws2812.led_count)
                    time.sleep(0.01)

        def pulsing_green():
            pass

        def red():
            pass

        def green():
            pass

        self.animation = animation_type
