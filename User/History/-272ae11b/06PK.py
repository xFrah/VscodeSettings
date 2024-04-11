import json
import os
from typing import Any, Dict

import requests


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
        with open(os.path.dirname(__file__) + "/config.json") as f:
            decoded = json.load(f)
        for key in keys:
            if key not in decoded:
                raise Exception(f"Missing {key} in configuration file")
        return decoded
    except Exception as e:
        print(e)
        debug("Configuration file not found or corrupted... Exiting.")
        exit(1)


def check_license() -> None:
    """
    Sends a request to the license server to check if the license is valid.
    """
    config = get_configuration()
    try:
        res = requests.get(
            f"http://license.lifesensor.cloud:8080/killswitch?terminal_id={config['terminal_id']}",
            timeout=5,
        )
        if res.status_code == 200:
            print("Killswitch OK")
        else:
            print("Killswitch NOT OK")
            exit(1)
    except Exception as e:
        print(e)


def ensure_receipts_folder():
    """
    Creates the receipts folder if it doesn't exist.
    """
    if not os.path.exists("receipts"):
        os.makedirs("receipts")
