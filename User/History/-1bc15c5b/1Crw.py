import time

import numpy
import vl53l5cx_ctypes as vl53l5cx
from matplotlib import cm

from helpers import flip_matrix
from PIL import Image


def tof_setup():
    print("[INFO] Configuring ToF:", end=" ", flush=True)
    vl53 = vl53l5cx.VL53L5CX()
    vl53.set_resolution(4 * 4)
    vl53.set_ranging_frequency_hz(60)
    vl53.set_integration_time_ms(10)
    vl53.start_ranging()
    print("Done.")
    return vl53


def absolute_diff(vector, base_vector, diff_threshold=2):
    """
    If the absolute difference between any two elements in the two vectors is greater than diff_threshold(default=2), return True, otherwise return False

    :param vector: the vector we're comparing to the base vector
    :param base_vector: The vector that we are comparing the other vectors to
    :param diff_threshold: The threshold for the absolute difference in centimeters
    :return: True or False
    """
    count = 0
    for index, (i, y) in enumerate(zip(vector, base_vector)):
        if index in [6, 7, 10, 11, 14, 15] and abs(i - y) > diff_threshold and (i != 0 and y != 0):
            count += 1
            if count > 3:
                return True
    return False


def render_tof(tof_frame):
    temp = numpy.array(tof_frame).reshape((4, 4))
    temp = [list(reversed(col)) for col in zip(*temp)]
    temp = flip_matrix(temp)
    arr = numpy.flipud(temp).astype('float64')

    # Scale view relative to the furthest distance
    # distance = arr.max()

    # Scale view to a fixed distance
    distance = 512

    # Scale and clip the result to 0-255
    arr *= (255.0 / distance)
    arr = numpy.clip(arr, 0, 255)

    # Force to int
    arr = arr.astype('uint8')

    # Convert to a palette type image
    img = Image.frombytes("P", (4, 4), arr)
    img.putpalette(pal)
    img = img.convert("RGB")
    img = img.resize((240, 240), resample=Image.NEAREST)
    img = numpy.array(img)
    return img


def get_palette(name):
    cmap = cm.get_cmap(name, 256)

    try:
        colors = cmap.colors
    except AttributeError:
        colors = numpy.array([cmap(i) for i in range(256)], dtype=float)

    arr = numpy.array(colors * 255).astype('uint8')
    arr = arr.reshape((16, 16, 4))
    arr = arr[:, :, 0:3]
    return arr.tobytes()


pal = get_palette("plasma")
