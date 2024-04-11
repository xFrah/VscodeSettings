import _thread
from machine import Pin
from adxl345_lib import adxl345
import time


class ACC:
    def __init__(self):
        scl = Pin(25)
        sda = Pin(26)
        cs = Pin(33, Pin.OUT)
        self._acc = adxl345(scl, sda, cs)
          // Send all interrrupts to INT 1 (PIN 2)
          writeRegister(INT_MAP,0);
          // Enable interrupts for INACTIVITY only.
          writeRegister(INT_ENABLE, 0x08);
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
