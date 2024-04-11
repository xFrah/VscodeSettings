# Description: Helper functions for the main.py file

import machine


def reset_with_timestamp():
    """Write to file current time and a number then reset the device"""
    # check if file already exists
    try:
        with open("last_reset.txt", "r") as f:
            n = int(f.read().split(",")[1])
    except Exception as e:
        n = 0
    with open("last_reset.txt", "w") as f:
        f.write("{},{}".format(time.time(), n + 1))
    machine.reset()


def parse_reset_file():
    """Parse the reset file and return the time and reset cause"""
    try:
        with open("last_reset.txt", "r") as f:
            str_time, n = f.read().split(",")
            # convert to time.time format
            str_time = float(str_time)
    except Exception as e:
        print("[HELPERS] Error parsing reset file:", e)
        return None, None
