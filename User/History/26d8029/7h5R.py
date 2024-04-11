import micropython
import helpers

micropython.alloc_emergency_exception_buf(100)
print("[BOOT] Emergency exception buffer allocated.")


def get_opmode() -> int:
    cases = {
        machine.PWRON_RESET: 0,
        machine.HARD_RESET: 0,
        machine.DEEPSLEEP_RESET: 1,
    }
    try:
        return cases[machine.reset_cause()]
    except KeyError:
        return 0


def detect_bootloop():
    timestamp, n = helpers.parse_reset_file()
    # if timestamp is less than 10 seconds ago and n is greater than 5

    if timestamp is not None and time.time() - timestamp < 10 and n > 5:
        print("[BOOT] Bootloop detected, resetting with timestamp.")
        # go to repl
        
