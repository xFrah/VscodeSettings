import micropython

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
