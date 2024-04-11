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
    folder_path = os.path.dirname(__file__) + "/receipts"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_receipt(receipt: dict, transaction_id: int) -> None:
    """
    Saves the receipt to the database.
    """
    ensure_receipts_folder()
    folder_path = os.path.dirname(__file__) + "/receipts"
    with open(f"{folder_path}/{transaction_id}.json", "w") as f:
        json.dump(receipt, f)

def get_receipt(transaction_id: int) -> dict:
    """
    Gets the receipt from the database.
    """
    ensure_receipts_folder()
    with open(f"receipts/{transaction_id}.json", "r") as f:
        return json.load(f)
