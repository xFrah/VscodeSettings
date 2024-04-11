from lib_ws2812 import NeoPixel
import time
import _thread


class LEDs:
    def __init__(self, bus=1, count=2, intensity=1.0):
        sp = machine.SPI(bus)
        sp.init(baudrate=3200000)
        self.ws2812 = NeoPixel(sp, count)
        self.animation = None
        self.intensity = intensity

    def thread(self):
        while True:
            try:
                while self.animation == "pulsing_red":
                    for i in range(100, 255):
                        self.ws2812[:] = [(i, 0, 0)]
                        time.sleep(0.01)
                    for i in range(255, 100, -1):
                        self.ws2812[:] = [(i, 0, 0)]
                        time.sleep(0.01)
                while self.animation == "pulsing_green":
                    for i in range(100, 255):
                        self.ws2812[:] = [(0, i, 0)]
                        time.sleep(0.01)
                    for i in range(255, 100, -1):
                        self.ws2812[:] = [(0, i, 0)]
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
