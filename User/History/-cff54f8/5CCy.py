import lib_pn532 as nfc
import _thread
from machine import Pin, SPI
from auth import auth
import time


class NFC:
    def __init__(self):
        # SPI
        self.spi_dev = SPI(2, baudrate=1000000, mosi=Pin(23), miso=Pin(19), sck=Pin(18))
        self.cs = Pin(5, Pin.OUT)
        self.cs.on()

        # SENSOR INIT
        self.pn532 = nfc.PN532(self.spi_dev, self.cs)
        ic, ver, rev, support = self.pn532.get_firmware_version()
        print("[NFC] Found PN532 with firmware version: {0}.{1}".format(ver, rev))

        # Configure PN532 to communicate with MiFare cards
        self.pn532.SAM_configuration()
        _thread.start_new_thread(self.read_nfc, ())

    def read_nfc(self, timeout=500):
        """Accepts a device and a timeout in millisecs"""
        while True:
            try:
                uid = self.pn532.read_passive_target(timeout=timeout)
            except Exception as e:
                print(e)
                uid = None
                time.sleep(1)
            if uid is not None:
                numbers = [i for i in uid]
                string_ID = "{}-{}-{}-{}".format(*numbers)
                # print("Found card with UID:", [hex(i) for i in uid])
                print("[NFC] Number_id: {}".format(string_ID))
                auth(string_ID)
