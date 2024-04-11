# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2023 Jose D. Montoya
#
# SPDX-License-Identifier: MIT

import time
from math import atan2, degrees
from machine import Pin, I2C
from lib_magnetometer import LIS3MDL
import _thread


def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


class Magnetometer:
    def __init__(self, buffer_size: int = 5):
        i2c = I2C(sda=Pin(26), scl=Pin(25))  # Correct I2C pins for RP2040
        self.lis = LIS3MDL(i2c)
        self.buffer_size = buffer_size
        self.buffer: list[tuple] = []
        self.calibrating = False
        self.displaced = False
        self.default_vector, self.default_vector_std = self.load_default_vector()

    def step(self):
        if self.calibrating:
            print("[MAGNETOMETER] Calibrating...")
        if not self.default_vector:
            print("[MAGNETOMETER] Default vector not loaded.")
        self.buffer.append(self.lis.magnetic)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get_heading(self):
        if not self.buffer:
            return None
        x, y, z = 0, 0, 0
        for _x, _y, _z in self.buffer:
            x += _x
            y += _y
            z += _z
        x /= len(self.buffer)
        y /= len(self.buffer)
        z /= len(self.buffer)
        return x, y, z

    def calibrate_default_vector(self):
        def mini_thread():
            vectors = []
            self.calibrating = True
            start = time.time()
            while time.time() - start < 10:
                vectors.append(self.lis.magnetic)
                time.sleep(0.1)
            # compute mean and standard deviation of every dimension
            x, y, z = 0, 0, 0
            for _x, _y, _z in vectors:
                x += _x
                y += _y
                z += _z
            x /= len(vectors)
            y /= len(vectors)
            z /= len(vectors)
            std_x, std_y, std_z = 0, 0, 0
            for _x, _y, _z in vectors:
                std_x += (_x - x) ** 2
                std_y += (_y - y) ** 2
                std_z += (_z - z) ** 2
            std_x /= len(vectors)
            std_y /= len(vectors)
            std_z /= len(vectors)
            std_x = std_x**0.5
            std_y = std_y**0.5
            std_z = std_z**0.5
            self.default_vector = (x, y, z)
            self.default_vector_std = (std_x, std_y, std_z)
            self.save_default_vector(self.default_vector, self.default_vector_std)
            self.calibrating = False
            print("[MAGNETOMETER] Calibrated default vector:", self.default_vector)

        _thread.start_new_thread(mini_thread, ())

    def load_default_vector(self):
        try:
            with open("default_vector.txt", "r") as f:
                vector = f.read().split(",")
                vector = [float(x) for x in vector]
                return vector[:3], vector[3:]
        except Exception as e:
            print("[MAGNETOMETER] Error loading default vector", e)
            return None, None

    def save_default_vector(self, vector, std_vector):
        # in file are 6 values, 3 for vector and 3 for std
        with open("default_vector.txt", "w") as f:
            f.write(",".join([str(x) for x in vector] + [str(x) for x in std_vector]))
        print("[MAGNETOMETER] Default vector saved.")
