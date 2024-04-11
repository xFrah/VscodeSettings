import _thread
from machine import Pin
from adxl345_lib import adxl345


class ACC:
    def __init__(self):
        scl = Pin(26)
        sda = Pin(25)
        cs = Pin(33, Pin.OUT)
        self._acc = adxl345(scl, sda, cs)
        _thread.start_new_thread(self.read, ())

    def read(self):
        while True:
            try:
                x, y, z = self._acc.readXYZ()
            except Exception as e:
                print(e)
                x, y, z = 0, 0, 0
            print("x:", x, "y:", y, "z:", z, "uint:mg")
            time.sleep(0.5)
