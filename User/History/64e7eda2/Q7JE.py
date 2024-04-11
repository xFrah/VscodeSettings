#!/usr/bin/env python3
import json
import os
import time

import numpy

import threading
import datetime

import helpers
from camera_utils import Camera
from edgetpu_utils import inference, setup_edgetpu
from new_led_utils import LEDs
from mog_utils import MOG
from tof_utils import tof_setup, render_tof
from watchdog import ping, pings, ignore
import cv2 as cv

print("[INFO] Starting...")

valid = {"name": str, "bin_id": int, "current_class": str, "bin_height": int, "bin_threshold": int}
config_and_data = {
    "target_distance": 150,
}


def watchdog_thread():
    """
    It checks if the threads are still alive. If they're not, it kills the program...
    """
    while True:
        time.sleep(5)
        for key, value in pings.items():
            if (datetime.datetime.now() - value).total_seconds() > 70 and key not in ignore:
                print(f"[ERROR] Thread [{key.getName()}] is not responding, killing...")
                helpers.kill()
            elif not key.is_alive():
                print(f"[ERROR] Thread [{key.getName()}] is not responding, killing...")
                helpers.kill()


def show_results(tof_frame, camera_frame, diff, cropped=None):
    """
    Displays things on the screen

    :param tof_frame: the depth frame from the ToF camera
    :param camera_frame: the current frame from the camera
    :param diff: the difference between the current frame and the background
    :param cropped: the cropped image
    """
    # render_tof(tof_frame)

    # cv.imshow("Diff", thresh)
    # cv.imshow("Cropped", cropped)
    cv.imshow("Camera", camera_frame)
    cv.imshow("Diff", diff)
    cv.waitKey(1) & 0xFF


def get_segmentation(camera_target_frame, background, camera_target_frame_index, buffer):
    rect, diff = helpers.get_diff(camera_target_frame, background)
    buffer_indexes = sorted(buffer.values(), key=lambda d: d[1])
    original = camera_target_frame_index + 0
    while not helpers.is_rect_good(rect, background):
        camera_target_frame_index += 1
        if camera_target_frame_index == len(buffer_indexes):
            print("[ERROR] No good frame found, skipping")
            rect = None
            break
        camera_target_frame = buffer_indexes[camera_target_frame_index][0]
        rect, diff = helpers.get_diff(camera_target_frame, background)
    print(f"[INFO] Original index: {original}, Revised index: {camera_target_frame_index}")
    return rect, diff


def get_frame_at_distance(
    tof_buffer: dict[datetime.datetime, tuple[numpy.array, float]],
    cap_buffer: dict[datetime.datetime, tuple[numpy.array, int]],
    distance: int,
):
    """
    Takes a buffer of frames, a buffer of distances and a target distance. Returns the frame and distance that are closest to the target distance.

    :param tof_buffer: A dictionary of time: full_matrix, distance
    :param cap_buffer: A dictionary of time: frame, frame_number
    :param distance: the distance in mm that you want to capture
    :return: The full matrix of the closest distance and the frame number of the closest frame
    :type tof_buffer: dict[datetime.datetime, tuple[numpy.array, float]]
    :type cap_buffer: dict[datetime.datetime, tuple[numpy.array, int]]
    :type distance: int
    """

    time_target_item = min(tof_buffer.items(), key=lambda d: abs(d[1][1] - distance))
    closest_frame_item = min(cap_buffer.items(), key=lambda d: abs((d[0] - time_target_item[0]).total_seconds()))
    print(f"[INFO] Target is frame {closest_frame_item[1][1]} at {time_target_item[1][1]}mm")
    print(f"[INFO] Distances: {[(round(dist[0].microsecond / 1000, 2), dist[1][1]) for dist in tof_buffer.items()]}")
    print(f"[INFO] Frames: {[(round(frame[0].microsecond / 1000, 2), frame[1][1]) for frame in cap_buffer.items()]}")
    print(f"[INFO] Time distance: {round(abs(time_target_item[0] - closest_frame_item[0]).total_seconds() * 1000, 2)}ms")
    return time_target_item[1][0], closest_frame_item[1][0], closest_frame_item[1][1]


def setup():
    """
    It sets up the camera, the LED strip, the VL53L0X sensor, the MQTT client, the TensorFlow interpreter, and the data manager
    :return: leds, interpreter, camera, vl53, initial_background, empty_tof_buffer, datamanager
    """

    leds = LEDs()
    mog = MOG()
    interpreter = setup_edgetpu()
    camera = Camera(leds, fast_mode=True)
    vl53 = tof_setup()
    tof_buffer = {}
    leds.stop_loading_animation()
    while leds.in_use():
        time.sleep(0.1)
    print("[INFO] Setup complete!")
    background = camera.grab_background(return_to_black=False)
    print("[INFO] Background grabbed!")
    for _ in range(150):
        if vl53.data_ready():
            data = vl53.get_data()
            mog.forward(data.distance_mm[0])
    leds.change_to_green()
    leds.black_from_green()
    threading.Thread(target=watchdog_thread, daemon=True, name="Watchdog").start()
    return leds, interpreter, camera, vl53, background, tof_buffer, mog


def main():
    leds, interpreter, camera, vl53, background, tof_buffer, mog = setup()
    thread = threading.current_thread()
    thread.setName("Main")
    print(f'[INFO] Main thread "{thread}" started.')
    count = 0
    movement = False
    start = datetime.datetime.now()
    last = datetime.datetime.now()
    print("[INFO] Ready for action!")
    while True:
        if vl53.data_ready():
            data = vl53.get_data().distance_mm[0]
            asd = mog.forward(data, display=True)
            if not movement:
                if len(asd) > 0 and (datetime.datetime.now() - start).total_seconds() > 1:
                    camera.shoot()
                    tof_buffer = {datetime.datetime.now(): (data, sum(asd) / len(asd))}
                    movement = True
                    print("[INFO] Movement detected")
                    start = datetime.datetime.now()
                    count = 1
            else:
                if len(asd) == 0 and ((now := datetime.datetime.now()) - start).total_seconds() > 0.3:
                    movement = False
                    buffer = camera.stop_shooting()
                    imgcopy = None
                    if not buffer:
                        print("[ERROR] No frames captured or broken session")
                        count = 0
                        buffer.clear()
                        tof_buffer.clear()
                        continue

                    print(f"[INFO] Stopped, FPS: {(count / (now - start).total_seconds(), len(buffer) / (now - start).total_seconds())}")

                    tof_target_frame, camera_target_frame, camera_target_frame_index = get_frame_at_distance(
                        tof_buffer, buffer, config_and_data["target_distance"]
                    )
                    rect, diff = get_segmentation(camera_target_frame, background, buffer)
                    if (rect is not None) and (diff is not None):
                        x, y, w, h = rect
                        imgcopy = camera_target_frame.copy()
                        cropped = imgcopy[y : y + h, x : x + w]
                        cv.rectangle(imgcopy, (x, y), (x + w - 1, y + h - 1), 255, 2)

                        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                            try:
                                cropped = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
                            except:
                                print("[ERROR] Cropped image is not a valid image")
                                count = 0
                                buffer.clear()
                                tof_buffer.clear()
                                print("[INFO] Waiting for movement...")
                                continue

                            label, score = inference(cropped, interpreter)
                            print(f"[INFO] Class: {label}, score: {int(score * 100)}%")

                            show_results(tof_target_frame, imgcopy, diff, cropped=cropped)

                            # leds.change_to_white()
                            # background = camera.grab_background(return_to_black=False)
                            # leds.black_from_white()
                            current_class = "paper"
                            if label == current_class:
                                leds.change_to_green()
                            else:
                                leds.change_to_red()
                            background = camera.grab_background(return_to_black=False)
                            if label == current_class:
                                leds.black_from_green()
                            else:
                                leds.black_from_red()

                    else:
                        print("[INFO] Object not found.")
                        show_results(tof_target_frame, camera_target_frame, diff)
                        background = camera.grab_background(return_to_black=True)

                    count = 0
                    buffer.clear()
                    tof_buffer.clear()
                    print("[INFO] Waiting for movement...")
                    last = datetime.datetime.now()
                else:
                    if len(asd) > 0:
                        tof_buffer[datetime.datetime.now()] = (data, sum(asd) / len(asd))
                    count += 1
            ping(thread)

        time.sleep(0.003)  # Avoid polling *too* fast


if __name__ == "__main__":
    main()
