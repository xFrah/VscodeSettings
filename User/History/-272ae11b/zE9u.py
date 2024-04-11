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
    try:
        with open("configuration.json") as f:
            decoded = json.load(f)
            # check if it has all the keys
            keys = ["pos_port", "printer_port", ]
            return json.load(f)
    except Exception as e:
        print(e)
        debug("Configuration file not found or corrupted...")
        exit(1)
