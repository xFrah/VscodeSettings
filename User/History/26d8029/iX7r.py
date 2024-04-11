import micropython
import helpers
import time
import os

micropython.alloc_emergency_exception_buf(100)
print("[BOOT] Emergency exception buffer allocated.")


class ExitToREPL(Exception):
    pass


def detect_bootloop():
    timestamp, n = helpers.parse_reset_file()
    # get how many seconds ago was the last reset
    if timestamp is None:
        return print("[BOOT] No visible resets.")

    ago = time.time() - timestamp
    print(f"[BOOT] Last reset was {ago:.2f} seconds ago, {n} resets in total.")

    if timestamp and ago < 30 and n > 5:
        print("[BOOT] Bootloop detected. Exiting to REPL.")
        helpers.remove("last_reset.txt")
        return True
    # if it was more than 30 seconds ago, delete the file
    elif timestamp and ago > 30:
        print("[BOOT] More than 30 seconds ago, deleting the file.")
        helpers.remove("last_reset.txt")


if detect_bootloop() or not helpers.get_configuration(raises=False):
    import ble_client

    while True:
        time.sleep(1)
