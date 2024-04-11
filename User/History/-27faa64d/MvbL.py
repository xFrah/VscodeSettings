import json
import os
import uasyncio as asyncio
import _thread

file_update_lock = _thread.allocate_lock()
wifi_credentials_lock = _thread.allocate_lock()


async def auth(idd):

    # check if lock is already acquired
    with file_update_lock:
        auth_dict = json.load(open("records_file.json", "r"))
    print(auth_dict)
    if idd in auth_dict.keys():
        return True
    return False


def get_wifi_credentials():
    with wifi_credentials_lock:
        try:
            wifi_dict = json.load(open("wifi.json", "r"))
        except Exception as e:
            print(e)
            return None, None
        # check if malformed
        if "ssid" not in wifi_dict.keys() or "password" not in wifi_dict.keys():
            return None, None
        return wifi_dict["ssid"], wifi_dict["password"]


def write_wifi_credentials(ssid, password):
    with wifi_credentials_lock:
        try:
            json.dump({"ssid": ssid, "password": password}, open("wifi.json", "w"))
        except Exception as e:
            print(e)
            return False
    return True
