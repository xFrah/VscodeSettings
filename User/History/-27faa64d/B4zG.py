import json
import os
import uasyncio as asyncio
import _thread

file_update_lock = _thread.allocate_lock()
wifi_credentials_lock = _thread.allocate_lock()


async def auth(idd):
    # open json file, get the dict and check if id is in key

    # check if lock is already acquired
    with file_update_lock:
        auth_dict = json.load(open("auth.json", "r"))
    print(auth_dict)
    if idd in auth_dict.keys():
        return True
    return False


def get_wifi_credentials():
    with wifi_credentials_lock:
        wifi_dict = json.load(open("wifi.json", "r"))
    return wifi_dict
