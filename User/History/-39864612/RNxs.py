# Description: Helper functions for the main.py file

import machine
import time
import bluetooth
from hashlib import sha256
import os


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
            return float(str_time), int(n)
    except Exception as e:
        return None, None


def get_mac_address() -> str:
    """
    Get MAC address of the device

    Returns:
        str: MAC address
    """
    mac = "".join(["{:02X}".format(x) for x in machine.unique_id()])
    return ":".join([mac[i : i + 2] for i in range(0, len(mac), 2)])


def get_ble_mac() -> str:

    ble = bluetooth.BLE()
    already_active = ble.active()
    if not already_active:
        ble.active(True)
        mac_bytes = ble.config("mac")[1]
        ble.active(False)
    else:
        mac_bytes = ble.config("mac")[1]

    # format it with : and uppercase
    mac = ":".join("{:02X}".format(b) for b in mac_bytes)
    return mac


def remove(path: str):
    try:
        os.remove(path)
    except Exception as e:
        print("Error deleting the file:", e)


def rename(old: str, new: str):
    try:
        os.rename(old, new)
    except Exception as e:
        print("Error renaming the file:", e)


def get_sha256(file) -> str:
    """
    Returns the sha256 hash of the specified file.

    Args:
        file (str): The file to get the sha256 hash of.

    Returns:
        str: The sha256 hash of the specified file.
    """

    sha256_hash = sha256()
    with open(file, "rb") as f:
        while True:
            byte_block = f.read(1024)
            if not byte_block:
                break
            sha256_hash.update(byte_block)
    hash_digest = sha256_hash.digest()
    hex_digest = "".join("{:02x}".format(byte) for byte in hash_digest)
    return hex_digest


def get_btree_sha256() -> str:
    """
    Returns the sha256 hash of the specified btree.
    """
    try:
        with open("badge_hashes", "r+b") as f:
            db = btree.open(f)
            sha256_hash = sha256()
            for value in db:
                sha256_hash.update(value)
            hash_digest = sha256_hash.digest()
            hex_digest = "".join("{:02x}".format(byte) for byte in hash_digest)
            return hex_digest
    except Exception as e:
        print("[UPDATER] Error opening database:", e)
        try:
            db.close()
        except Exception:
            pass
        return "-1"


def get_last_update_info() -> dict:
    """
    Returns the last update info dictionary.
    """
    try:
        with open("update_info.json", "r") as f:
            thing = json.loads(f.read())
            # check that local_time_int is of int type
            thing["local_time_int"] = int(thing["local_time_int"])
            return thing
    except Exception as e:
        print("[UPDATER] Error reading update_info.json:", e)
        remove("update_info.json")
        return None


def get_configuration(raises: bool = True) -> dict:
    """
    Returns the configuration dictionary.
    """
    try:
        with open("config.json", "r") as f:
            return json.loads(f.read())
    except Exception as e:
        print("[CONFIG] Error reading config file:", e)
        remove("config.json")
        if raises:
            raise e


def set_configuration(config: dict):
    """
    Sets the configuration dictionary.
    """
    try:
        with open("config.json", "w") as f:
            f.write(json.dumps(config))
    except Exception as e:
        print("[CONFIG] Error writing config file:", e)


def set_configuration_item(key: str, value):
    """
    Sets the configuration dictionary.
    """
    config = get_configuration()
    config[key] = value
    set_configuration(config)
