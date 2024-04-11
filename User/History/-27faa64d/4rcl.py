import json
import os
import uasyncio as asyncio
from threading import Lock

file_update_lock = Lock()


async def auth(idd):
    # open json file, get the dict and check if id is in key

    # check if lock is already acquired
    if not file_update_lock.locked():
        with file_update_lock:
            auth_dict = json.load(open("auth.json", "r"))
        print(auth_dict)
        if idd in auth_dict.keys():
            return True
        return False
    return True
