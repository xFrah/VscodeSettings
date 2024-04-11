import json


def debug(msg: str):
    """
    Prints a debug message if debug_flag is True.
    """
    print(">", msg)


def get_configuration():
    """
    Returns the configuration dictionary.
    """
    with open("configuration.json") as f:
        return json.load(f)
