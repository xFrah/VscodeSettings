import micropython
import helpers

micropython.alloc_emergency_exception_buf(100)
print("[BOOT] Emergency exception buffer allocated.")


class ExitToREPL(Exception):
    pass


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
    # get how many seconds ago was the last reset
    ago = time.time() - timestamp
    print(f"[BOOT] Last reset was {ago:.2f} seconds ago, {n} resets in total.")

    if timestamp is not None and time.time() - timestamp < 30 and n > 5:
        print("[BOOT] Bootloop detected. Exiting to REPL.")
        raise ExitToREPL


detect_bootloop()
