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
            keys = ["pos_port", "printer_port"]
            for key in keys:
                if key not in decoded:
                    raise Exception(f"Missing {key} in configuration file")
            return json.load(f)
    except Exception as e:
        print(e)
        debug("Configuration file not found or corrupted... Exiting.")
        exit(1)
