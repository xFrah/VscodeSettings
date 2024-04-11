import json
import os
import _thread

file_update_lock = _thread.allocate_lock()


def auth(idd):
    # check if lock is already acquired
    with file_update_lock:
        auth_dict = json.load(open("records_file.json", "r"))
    print(auth_dict)
    if "allowed" in auth_dict.keys() and idd in auth_dict["allowed"]:
        return True
    return False
