# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2023 Jose D. Montoya
#
# SPDX-License-Identifier: MIT

import time
import _thread
from math import atan2, degrees
from machine import Pin, I2C
from lib_magnetometer import LIS3MDL


class Magnetometer:
    def __init__(self):
        i2c = I2C(sda=Pin(26), scl=Pin(25))  # Correct I2C pins for RP2040
        self.lis = LIS3MDL(i2c)
        self.buffer: list[tuple] = []
        self.calibrating = False
        self.displaced = False
        self.default_vector = self.load_default_vector()
        # start thread reader
        _thread.start_new_thread(self.reader, ())

    def reader(self):
        while True:
            time.sleep(0.5)
            if self.calibrating:
                continue
            self.buffer.append(self.lis.magnetic)
            if len(self.buffer) > 5:
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

    def calibrate_default_vector():
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
            vector = (x, y, z)
            std_x, std_y, std_z = 0, 0, 0
            for _x, _y, _z in vectors:
                std_x += (_x - x) ** 2
                std_y += (_y - y) ** 2
                std_z += (_z - z) ** 2
            std_x /= len(vectors)
            std_y /= len(vectors)
            std_z /= len(vectors)
            std_x = std_x ** 0.5
            std_y = std_y ** 0.5
            std_z = std_z ** 0.5

            
            self.save_default_vector(vector)
            self.calibrating = False

        _thread.start_new_thread(mini_thread, ())

    def load_default_vector(self):
        with open("default_vector.txt", "r") as f:
            self.default_vector = tuple([float(x) for x in f.read().split(",")])

    def save_default_vector(self, vector):
        with open("default_vector.txt", "w") as f:
            f.write(",".join([str(x) for x in vector]))
