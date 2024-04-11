from helpers import reset_with_timestamp

import neopixel
import time
import _thread
import machine


class LEDs:
    def __init__(self, count=2, intensity=1.0):
        self.ws2812 = neopixel.NeoPixel(machine.Pin(22), count)
        self.animation = None
        self.intensity = intensity
        _thread.start_new_thread(self.thread, ())

    def thread(self):
        try:
            while True:
                print("[LED] Animation:", self.animation)
                try:
                    print("[LED] Displaying data...")
                    while self.animation == "pulsing_red":
                        for i in range(100, 255):
                            self.ws2812[:] = (i, 0, 0)
                            time.sleep(0.01)
                        for i in range(255, 100, -1):
                            self.ws2812[:] = (i, 0, 0)
                            time.sleep(0.01)
                        print("[LED] Begin Animation:", self.animation)
                    while self.animation == "pulsing_green":
                        for i in range(100, 255):
                            self.ws2812[:] = (0, i, 0)
                            time.sleep(0.01)
                        for i in range(255, 100, -1):
                            self.ws2812[:] = (0, i, 0)
                            time.sleep(0.01)
                    while self.animation == "red":
                        self.ws2812[:] = (255, 0, 0)
                        time.sleep(0.1)
                    while self.animation == "green":
                        self.ws2812[:] = (0, 255, 0)
                        time.sleep(0.1)
                    while self.animation is None:
                        self.ws2812[:] = (0, 0, 0)
                        time.sleep(0.1)
                except Exception as e:
                    print("[LED] Error while displaying data", e)
                time.sleep(0.3)
        except Exception as e:
            print("[LED] Error in LED thread:", e)
            reset_with_timestamp()
