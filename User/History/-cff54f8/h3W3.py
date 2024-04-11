import lib_pn532 as nfc
import _thread
from machine import Pin, SPI
from auth import auth
import time


class NFC:
    def __init__(self, sec_obj):
        # SPI
        self.spi_dev = SPI(2, baudrate=1000000, mosi=Pin(23), miso=Pin(19), sck=Pin(18))
        self.cs = Pin(5, Pin.OUT)
        self.cs.on()
        self.sec_obj = sec_obj
        # SENSOR INIT
        self.pn532 = nfc.PN532(self.spi_dev, self.cs)
        for i in range(4):
            try:
                ic, ver, rev, support = self.pn532.get_firmware_version()  # TODO lib says this often fails first time.
            except RuntimeError as e:
                print(e)
                time.sleep(0.3)
        print("[NFC] Found PN532 with firmware version: {0}.{1}".format(ver, rev))

        # Configure PN532 to communicate with MiFare cards
        self.pn532.SAM_configuration()
        _thread.start_new_thread(self.read_nfc, ())

    def read_nfc(self, timeout=500):
