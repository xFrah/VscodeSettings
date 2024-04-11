import threading
import time
import cv2

import numpy
import serial
import pyudev
from usb_stuff import USB_Dispatcher

dim = 625


def decode(line):
    # TODO: Stop creating lists here
    checksum = line[20 + dim]
    summe = sum(line[: 20 + dim]) % 256
    if checksum != summe:
        return
    # stringa = f"{checksum:08b} {summe:08b}"
    image_ = line[20 : 20 + dim]
    try:
        arr = numpy.frombuffer(image_, numpy.uint8)
        arr = arr.reshape(25, 25)
    except Exception as e:
        print(e)
        return
    return arr


class Tof:
    def __init__(self, addr: list, usb=False, buffer_size: int = 5, baudrate=115200):
        self.addr = addr
        self.baudrate = baudrate
        # print("Tof >> Opening serial port: ", self.com)
        self.current_addr = 0
        self.serial_tof: serial.Serial = serial.Serial(
            self.addr[self.current_addr],
            self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self.read_thread: threading.Thread = threading.Thread(target=self._read_data)
        self.buffer_size = buffer_size
        self.data = bytes()
        self.images = []
        self.reading = True
        if usb:
            self.serial_tof.write(b"AT+DISP=3\r")
        else:
            self.serial_tof.write(b"AT+DISP=5\r")
        self.serial_tof.write(b"AT+FPS=15\r")
        self.serial_tof.write(b"AT+BAUD=2\r")
        self.serial_tof.write(b"AT+UNIT=10\r")
        self.serial_tof.write(b"AT+BINN=4\r")
        self.serial_tof.write(b"AT+SAVE\r")
        self.read_thread.start()

    def _read_data(self):
        last = time.time()
        start = time.time()
        c = 0
        while True:
            # get number of bytes that are ready to be read by serial
            to_add = self.serial_tof.read_all()
            if not to_add:
                time.sleep(0.05)
                if last < time.time() - 1:
                    self.reload_serial_connection()
                    print(f"Tof-{self.addr} >> No data for 1 second, turning to {self.addr[self.current_addr]}")
                    last = time.time()
                continue
            last = time.time()
            self.data += to_add
            # print to add
            # print(f"Tof >> Read {len(to_add)} bytes, total: {len(self.data)}")
            if len(self.data) == 0 or len(self.data) < dim - 50:
                time.sleep(0.05)
                continue

            # start = self.data.find(b"\x00\xff\x20\x27")
            start = self.data.find(b"\x00\xff")
            if start == -1:
                print("Tof >> Header not found, len: ", str(len(self.data)))
                self.data = bytes()
            else:
                self.data = self.data[start:]
                # TODO: Replace this with del self.data[:start]
                # print(f"Removed {start} ")
                end = 22 + dim
                if len(self.data) >= end:
                    data_to_decode = self.data[:end]
                    image = decode(data_to_decode)
                    if image is not None:
                        self.images.append(image)
                        c += 1
                        if c == 10:
                            # print(f"Tof >> FPS: {c / (time.time() - start)}")
                            start = time.time()
                            c = 0
                    self.images[: -self.buffer_size] = []

                    # check if self.data has index end + 1
                    self.data = self.data[end:]
                    # TODO: Replace this with del self.data[:end]
                    # print(f"Removed {end + 2} bytes")

    def get_next_addr(self):
        self.current_addr += 1
        if self.current_addr >= len(self.addr):
            self.current_addr = 0
        return self.addr[self.current_addr]

    def get_image(self, pop: bool = True):
        # print("Tof >> Getting image, buffer size: ", len(self.images))
        return (list.pop if pop else list.__getitem__)(self.images, 0) if len(self.images) else None

    def fix_buffer_leak(self):
        if len(self.images) > self.buffer_size:
            self.images[: -self.buffer_size] = []
            print("Tof >> Buffer leak fixed, buffer size: ", len(self.images))

    def active(self, is_reading: bool):
        if not isinstance(is_reading, bool):
            return print("Tof >> Invalid type, active takes a bool as argument.")
        self.reading = is_reading

    def reload_serial_connection(self):
        self.serial_tof.close()
        self.serial_tof = serial.Serial(
            self.get_next_addr(),
            self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )

    def close(self):
        self.serial_tof.close()


if __name__ == "__main__":
    image = None
    image_buffer = []
    c = 0
    start = time.time()
    usb_disp = USB_Dispatcher(pyudev.Context())
    addr = usb_disp.get_tty_by_devicename("TOF2")
    #addr = ["/dev/ttyUSB1", "/dev/ttyUSB2"]
    print("Tof >> Serial port address: ", addr)
    tof = Tof(addr=addr, usb=True, buffer_size=30)
    while True:
        if tof.images:
            # print("Tof >> Image in buffer")
            # get numpy array from bytes
            image_t = tof.get_image()
            if image_t is None:
                print("Invalid image")
            else:
                image = image_t
                c += 1
                if c % 10 == 0:
                    print(f"FPS: {c / (time.time() - start)}")
                    c = 0
                    start = time.time()
        if image is not None:
            cv2.imshow("image", image)
            cv2.waitKey(1)
