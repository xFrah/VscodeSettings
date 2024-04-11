class Acc:
    def __init__(self):
        self.scl = Pin(26)
        sda = Pin(25)
        cs = Pin(33, Pin.OUT)
        snsr = adxl345(scl, sda, cs)
        while True:
            x, y, z = snsr.readXYZ()
            print("x:", x, "y:", y, "z:", z, "uint:mg")
            time.sleep(0.5)
