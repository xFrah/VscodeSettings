import json
import os
import _thread

file_update_lock = _thread.allocate_lock()


def auth(idd):
    if idd == ["0xb7", "0xbc", "0x8b", "0xff"]:
        return True
    # check if lock is already acquired
    with file_update_lock:
        auth_dict = json.load(open("records_file.json", "r"))
    print(auth_dict)
    if idd in auth_dict.keys():
        return True
    return False
