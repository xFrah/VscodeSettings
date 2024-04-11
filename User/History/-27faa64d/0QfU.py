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
