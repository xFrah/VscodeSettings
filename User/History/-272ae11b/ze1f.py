import json
import os
from typing import Any, Dict


def debug(msg: str):
    """
    Prints a debug message if debug_flag is True.
    """
    print(">", msg)


def get_configuration() -> Dict[str, Any]:
    """
    Returns the configuration dictionary.
    """
    keys = ["pos_port", "printer_port"]
    try:
        with open(os.path.dirname(__file__) "config.json") as f:
            decoded = json.load(f)
        for key in keys:
            if key not in decoded:
                raise Exception(f"Missing {key} in configuration file")
        return decoded
    except Exception as e:
        print(e)
        debug("Configuration file not found or corrupted... Exiting.")
        exit(1)
