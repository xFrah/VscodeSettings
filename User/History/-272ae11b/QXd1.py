def debug(msg: str):
    """
    Prints a debug message if debug_flag is True.
    """
    if debug_flag:
        print(">", msg)
