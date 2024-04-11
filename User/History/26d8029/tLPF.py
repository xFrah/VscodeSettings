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

    if timestamp is not None and time.time() - timestamp < 30 and n > 5:
        print("[BOOT] Bootloop detected. Exiting to REPL.")
        try:
            os.remove("last_reset.txt")
        except Exception as e:
            print("[BOOT] Error deleting the file: {}".format(e))
        return True
    # if it was more than 30 seconds ago, delete the file
    elif timestamp is not None and time.time() - timestamp > 30:
        print("[BOOT] More than 30 seconds ago, deleting the file.")
        try:
            os.remove("last_reset.txt")
        except Exception as e:
            print("[BOOT] Error deleting the file: {}".format(e))


if detect_bootloop():
    import ble_client

