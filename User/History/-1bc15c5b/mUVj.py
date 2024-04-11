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
