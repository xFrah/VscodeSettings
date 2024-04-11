import argparse
import threading
import time

import cv2
import numpy
import serial
from matplotlib import pyplot as plt


def decode(line):
    # TODO: Stop creating lists here
    checksum = line[10020]
    summe = sum(line[:10020]) % 256
    if checksum != summe:
        return
    # stringa = f"{checksum:08b} {summe:08b}"
    image_ = line[20:10020]
    try:
        arr = numpy.frombuffer(image_, numpy.uint8)
        arr = arr.reshape(100, 100)
    except ValueError:
        return
    return arr


class Tof:
    def __init__(self, addr: list, name, usb_manager, buffer_size: int = 5, background_shape=(100, 100), matrix_path=None, baudrate=921600):
        self.addr = addr
        self.name = name
        self.usb_manager = usb_manager
        self.baudrate = baudrate
        # print("Tof >> Opening serial port: ", self.com)
        self.current_addr = 0
        self.serial_tof: serial.Serial = serial.Serial(self.addr[self.current_addr], self.baudrate, rtscts=True, dsrdtr=True)
        self.read_thread: threading.Thread = threading.Thread(target=self._read_data)
        self.buffer_size = buffer_size
        self.background_shape = background_shape
        self.background_padding = numpy.zeros((background_shape[0], background_shape[1], 4), numpy.uint8)
        self.background_padding_gray = numpy.zeros((background_shape[0], background_shape[1]), numpy.uint8)
        self.data = bytes()
        self.images = []
        self.reading = True
        self.colormap = plt.get_cmap("rainbow")
        # load matrix with numpy at path matrix_path
        self.M = numpy.load(matrix_path) if matrix_path else None
        self.M_inv = None
        # send at command to start streaming
        self.serial_tof.write(b"AT+DISP=3\r")
        self.serial_tof.write(b"AT+FPS=15\r")
        self.serial_tof.write(b"AT+BAUD=5\r")
        self.read_thread.start()

    def reload_addresses(self):
        self.addr = self.usb_manager.get_tty_by_devicename(self.name)

    def _read_data(self):
        while True:
            print(f"Tof >> Reading data from {self.addr[self.current_addr]}")
            
            try:
                last = time.time()
                start = time.time()
                c = 0
                retries = 0
                while True:
                    # get number of bytes that are ready to be read by serial
                    to_add = self.serial_tof.read_all()
                    if not to_add:
                        time.sleep(0.02)
                        if last < time.time() - 1:
                            # This check is necessary because the usb reset could be coming from the other tof.
                            retries += 1
                            if retries > 5:
                                self.close()
                                self.usb_manager.reset(self.name)
                                self.reload_serial_connection()
                                print("Tof >> Too much retries, hard usb reset.")
                                retries = 0
                            else:
                                self.reload_serial_connection()
                                print(f"Tof-{self.addr} >> No data for 1 second, turning to {self.addr[self.current_addr]}")
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
                    time.sleep(0.01)
            except Exception as e:
                print("Tof >> Error in tof reading thread: ", e)

    def get_distance_at_cv2_rectangle(self, x: int, y: int, w: int, h: int) -> tuple or None:
        if not isinstance(x, int) or not isinstance(y, int) or not isinstance(w, int) or not isinstance(h, int):
            return print("Tof >> Invalid type, get_distance_at_cv2_rectangle takes 4 ints as arguments.")
        image_ = self.get_image(pop=False)

        if image_ is None:
            return None

        warp = self.warp_image(image_, gray=True)

        # sort the distances in rectangle and get the first half
        # dist_list = sorted(warp[y : y + h, x : x + w].flatten())[: int(w * h / 2)]
        median_dist = numpy.median(warp[y : y + h, x : x + w])
        return median_dist, warp

    def get_next_addr(self):
        self.current_addr += 1
        if self.current_addr >= len(self.addr):
            self.current_addr = 0
        return self.addr[self.current_addr]

    def change_warp_matrix(self, src: numpy.ndarray, dst: numpy.ndarray):
        self.M = cv2.getPerspectiveTransform(src, dst)
        # now find the matrix that would get the image back to normal
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def normalize_image(self, frame: numpy.ndarray, gray=False) -> numpy.ndarray:
        if gray:
            new_bg = self.background_padding_gray.copy()
            new_bg[:100, :100] = frame
        else:
            new_bg = self.background_padding.copy()
            new_bg[:100, :100, :] = frame
        return new_bg

    def color_image(self, frame: numpy.ndarray) -> numpy.ndarray:
        frame = self.colormap(frame / 255)[:, :, :3] * 255
        frame = frame.astype(numpy.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    def get_image(self, pop: bool = True):
        # print("Tof >> Getting image, buffer size: ", len(self.images))
        return (list.pop if pop else list.__getitem__)(self.images, 0) if len(self.images) else None

    def warp_image(self, image, gray=False) -> numpy.ndarray:
        image = self.normalize_image(image, gray=gray)
        if self.M is None:
            return image
        return cv2.warpPerspective(image, self.M, self.background_shape[::-1])

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
        self.serial_tof = serial.Serial(self.get_next_addr(), self.baudrate, rtscts=True, dsrdtr=True)

    def close(self):
        self.serial_tof.close()


image = None
image_buffer = []
c = 0
start = time.time()
if __name__ == "__main__":
    # parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="COM8", help="Serial port address")
    args = parser.parse_args()
    print("Tof >> Serial port address: ", args.addr)
    tof = Tof(addr=args.addr, buffer_size=30)
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
            image_buffer.append(image)
            # wait one second for key press
            key = cv2.waitKey(1)
            if key == ord("q"):
                # save all images in buffer
                for i, img in enumerate(image_buffer):
                    cv2.imwrite(f"image_{i}.png", img)
                break
