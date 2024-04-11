from helpers import reset_with_timestamp

from nfc import NFC
from lis3mdl import Magnetometer, euclidean_distance
import _thread
from auth import auth
import time
from leds import LEDs
import json
import machine
import btree


class __SecurityObject:
    def __init__(self, std_threshold_scale=5):
        self._closed = True
        self._authed = False
        self.std_threshold_scale = std_threshold_scale
        self.file_update_lock = _thread.allocate_lock()
        self.timestamp_opened = time.time()
        self.need_calibration = False
        self.nfc = NFC()
        print("[SEC] NFC setup complete.")
        self.mag = Magnetometer()
        print("[SEC] Magnetometer setup complete.")
        self.leds = LEDs()
        _thread.start_new_thread(self.security_thread, ())

    def security_thread(self):
        """Accepts a device and a timeout in millisecs"""
        print("[SEC] Security thread started.")
        print("[SEC] Calibrating magnetometer...")
        try:
            while True:
                if self.need_calibration and not self.mag.calibrating:
                    print("[SEC] Calibrating magnetometer...")
                    self.mag.calibrate_default_vector()
                    self.need_calibration = False
                uid = self.read_nfc()
                if uid is not None:
                    res = self.auth("{}-{}-{}-{}".format(*[i for i in uid]))
                    if res and time.time() - self.timestamp_opened > 10:
                        if self._authed:
                            self._authed = False
                            self.timestamp_opened = time.time()
                            print("[NFC] Closing auth session.")
                        else:
                            self._authed = True
                            self.timestamp_opened = time.time()
                            print("[NFC] Auth session begins.")
                else:
                    if self._authed and time.time() - self.timestamp_opened > 60:
                        self._authed = False
                        print("[NFC] Timeout, closing auth session.")

                if not self.mag.calibrating:
                    self.mag.step()
                    v = self.mag.get_heading()
                    if v and self.mag.default_vector:
                        # use self.mag.default_vector_std(std_x, std_y, std_z) to know if the device is displaced
                        if self.vector_compare(v, self.mag.default_vector, self.mag.default_vector_std, self.std_threshold_scale):
                            if not self._closed:
                                self._closed = True
                                print("[MAG] Device closed.")
                        else:
                            if self._closed:
                                self._closed = False
                                print("[MAG] Device opened.")
                    else:
                        print(v, self.mag.default_vector)

                if self.mag.calibrating:
                    self.leds.animation = "pulsing_blue"
                elif self._closed and not self._authed:
                    self.leds.animation = "red"
                elif self._closed and self._authed:
                    self.leds.animation = "green"
                elif not self._closed and not self._authed:
                    self.leds.animation = "pulsing_red"
                elif not self._closed and self._authed:
                    self.leds.animation = "pulsing_green"
                time.sleep(0.1)
        except Exception as e:
            print("[SEC] Error in security thread:", e)
            reset_with_timestamp()

    def auth(self, badge):
        return True
        try:
            # check if lock is already acquired
            with self.file_update_lock:
                f = open("badge_hashes", "r+b")
                db = btree.ope4n(f)
                authorized = badge in db
                db.close()
                f.close()
            return authorized
        except Exception as e:
            print("[SEC] Error opening database:", e)
            try:
                db.close()
                f.close()
            except Exception as e:
                print("[SEC] Error closing database:", e)
            return True

    def read_nfc(self, timeout=500):
        """Returns the tag that is currently in the NFC reader."""
        try:
            uid = self.nfc.pn532.read_passive_target(timeout=timeout)
        except Exception as e:
            return print(e)
        return uid

    def vector_compare(self, vec, root_vec, stds, scale=1):
        debug_output = []  # List to store the debug information for each component
        flag = True

        for i in range(3):
            lower_bound = root_vec[i] - scale * stds[i]
            upper_bound = root_vec[i] + scale * stds[i]
            difference = vec[i] - root_vec[i]

            # Store debug information
            debug_output.append(f"{i}: {difference:+.2f},[{scale * stds[i]:.2f}]")

            if not (lower_bound <= vec[i] <= upper_bound):
                # print(", ".join(debug_output))
                flag = False

        # print(", ".join(debug_output))
        return flag

    @property
    def closed(self):
        return self._closed

    @property
    def authed(self):
        return self._authed

    @closed.setter
    def closed(self, value):
        self._closed = value

    @authed.setter
    def authed(self, value):
        self._authed = value


security_object = __SecurityObject()
