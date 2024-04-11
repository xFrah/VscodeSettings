import threading
import time
import serial


def decode(line):
    # TODO: Stop creating lists here
    checksum = line[10020]
    summe = sum(line[:10020]) % 256
    if checksum != summe:
        return
    # stringa = f"{checksum:08b} {summe:08b}"
    image_ = line[20:10020]
    try:
        # reshape without using numpy
        arr = []
        for i in range(0, 10000, 100):
            arr.append(image_[i : i + 100])
    except Exception as e:
        print(e)
        return
    return arr


class Tof:
    def __init__(self, addr: str, buffer_size: int = 5, baudrate=115200):
        self.addr = addr
        self.baudrate = baudrate
        # print("Tof >> Opening serial port: ", self.com)
        self.current_addr = 0
        self.serial_tof = serial.Serial(self.addr, self.baudrate)
        self.serial_tof.init(self.baudrate)
        self.read_thread = Thread(self._read_data, ())
        self.buffer_size = buffer_size
        self.data = bytes()
        self.images = []
        self.reading = True
        # send at command to start streaming
        self.serial_tof.write(b"AT+DISP=3\r")
        self.serial_tof.write(b"AT+FPS=15\r")
        self.serial_tof.write(b"AT+BAUD=5\r")
        self.serial_tof.write(b"AT+UNIT=10\r")
        self.read_thread.start()

    def _read_data(self):
        last = time.time()
        while True:
            # get number of bytes that are ready to be read by serial
            to_add = self.serial_tof.read()
            if not to_add:
                time.sleep(0.02)
                if last < time.time() - 1:
                    self.reload_serial_connection()
                    print(f"Tof-{self.addr} >> No data for 1 second, reloading serial connection.")
                    last = time.time()
                continue
            last = time.time()
            self.data += to_add
            # print to add
            # print(f"Tof >> Read {len(to_add)} bytes, total: {len(self.data)}")
            if len(self.data) == 0 or len(self.data) < 24000:
                time.sleep(0.005)
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
                end = 10022
                if len(self.data) >= end:
                    data_to_decode = self.data[:end]
                    image = decode(data_to_decode)
                    if image is not None:
                        self.images.append(image)
                    self.images[: -self.buffer_size] = []

                    # check if self.data has index end + 1
                    self.data = self.data[end:]
                    # TODO: Replace this with del self.data[:end]
                    # print(f"Removed {end + 2} bytes")
            time.sleep(0.01)

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
        self.serial_tof: UART = UART(self.addr, self.baudrate)
        self.serial_tof.init(self.baudrate)

    def close(self):
        self.serial_tof.close()


image_ = None
image_buffer = []
c = 0
start_ = time.time()
tof = Tof(addr=0, buffer_size=30)
while True:
    if tof.images:
        # print("Tof >> Image in buffer")
        # get numpy array from bytes
        image_t = tof.get_image()
        if image_t is None:
            print("Invalid image")
        else:
            image_ = image_t
            c += 1
            if c % 10 == 0:
                print(f"FPS: {c / (time.time() - start_)}")
                c = 0
                start_ = time.time()
    else:
        time.sleep(0.01)
