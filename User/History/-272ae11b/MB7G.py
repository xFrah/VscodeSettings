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
        # check if configuration.json exists

        with open("configuration.json") as f:
            return json.load(f)
    except Exception as e:
        print(e)
        print("Configuration file not found")
        exit(1)
