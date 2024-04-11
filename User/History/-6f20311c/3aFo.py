import _thread


class Acc:
    def __init__(self):
        scl = Pin(26)
        sda = Pin(25)
        cs = Pin(33, Pin.OUT)
        self._acc = adxl345(scl, sda, cs)

    def read(self):
        while True:
            x, y, z = self._acc.readXYZ()
            print("x:", x, "y:", y, "z:", z, "uint:mg")
            time.sleep(0.5)
