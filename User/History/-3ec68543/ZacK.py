import os
import json


def check_if_valid_mac(mac):
    if len(mac) != 17:
        print(f"Invalid MAC address length {len(mac)}.")
        return False
    for i in range(2, 17, 3):
        if mac[i] != ":":
            print(f"No colon at position {i}.")
            return False
    return True


def check_files():
    if not os.path.isfile("aliases.json"):
        with open("aliases.json", "w") as f:
            f.write("{}")
        print("Created aliases.json.")
    if not os.path.isfile("kits.json"):
        with open("kits.json", "w") as f:
            f.write("{}")
        print("Created kits.json.")


def command_kit(args):
    # check if kits.json exists
    check_files()
    # read kits.json
    with open("kits.json", "r") as f:
        kits: dict[str, list[str]] = json.load(f)
    action, mac_or_alias = args
    if action == "add":
        if check_if_valid_mac(mac_or_alias):
            if mac_or_alias in kits:
                print(f"Kit {mac_or_alias} already exists.")
            else:
                kits[mac_or_alias] = list()
                print(f"Kit {mac_or_alias} added.")
        else:
            # invalid mac, no alias for adding
            print(f"Invalid MAC address {mac_or_alias}.")
    elif action == "remove":
        if check_if_valid_mac(mac_or_alias):
            mac = mac_or_alias
        else:
            mac = get_mac_from_alias(mac_or_alias)
        if mac in kits:
            del kits[mac_or_alias]
            # read aliases.json
            command_clearalias([mac])
            print(f"Kit {mac} removed.")
        else:
            print(f"Kit {mac if mac is not None else mac_or_alias} not found.")
    else:
        print(f"Unknown action {action}.")
