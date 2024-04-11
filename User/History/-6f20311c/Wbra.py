import _thread
from machine import Pin
from adxl345_lib import adxl345
from micropython import const
import time

INT_MAP = const(0x2F)
INT_ENABLE = const(0x2E)


class ACC:
    def __init__(self):
        scl = Pin(25)
        sda = Pin(26)
        cs = Pin(33, Pin.OUT)
        self._acc = adxl345(scl, sda, cs)
        self.setup_interrupts()
        _thread.start_new_thread(self.read, ())

    def read(self):
        if not self._acc.is_connected():
            print("[ADXL345] Can't read data, device is not connected.")
            return
        while True:
            try:
                x, y, z = self._acc.readXYZ()
            except Exception as e:
                print(e)
                x, y, z = 0, 0, 0
            # print("x:", x, "y:", y, "z:", z, "uint:mg")
            time.sleep(0.5)

    def setup_interrupts(self):
        # Send all interrrupts to INT 1 (PIN 2)
        self._acc.writeByte(INT_MAP, 0)
        # Enable interrupts for INACTIVITY only.
        self._acc.writeByte(INT_ENABLE, 0b01010000)
