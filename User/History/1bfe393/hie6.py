from lib_ws2812 import WS2812
import time
import _thread


class LEDs:
    def __init__(self, bus=1, count, intensity=1.0):
        self.ws2812 = WS2812(spi_bus=1, led_count=count)
        self.animation = None
        self.intensity = intensity

    def thread(self):
        while True:
            try:
                while self.animation == "pulsing_red":
                    for i in range(100, 255):
                        self.ws2812.show([(i, 0, 0)] * self.ws2812.led_count)
                        time.sleep(0.01)
                    for i in range(255, 100, -1):
                        self.ws2812.show([(i, 0, 0)] * self.ws2812.led_count)
                        time.sleep(0.01)
                while self.animation == "pulsing_green":
                    for i in range(100, 255):
                        self.ws2812.show([(0, i, 0)] * self.ws2812.led_count)
                        time.sleep(0.01)
                    for i in range(255, 100, -1):
                        self.ws2812.show([(0, i, 0)] * self.ws2812.led_count)
                        time.sleep(0.01)
                while self.animation == "pulsing_blue":
                    for i in range(100, 255):
                        self.ws2812.show([(0, 0, i)] * self.ws2812.led_count)
                        time.sleep(0.01)
                    for i in range(255, 100, -1):
                        self.ws2812.show([(0, 0, i)] * self.ws2812.led_count)
                        time.sleep(0.01)
            except Exception as e:
                print("[LED] Error while displaying data", e)
            time.sleep(0.3)
