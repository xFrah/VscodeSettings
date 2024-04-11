import os
import json


def command_alias(args: list[str]):
    # check if aliases.json exists
    check_files()
    # read aliases.json
    with open("aliases.json", "r") as f:
        aliases: dict[str, list[str]] = json.load(f)
    action, mac, alias = args
    if action == "add":
        # check if mac in kits.json
        with open("kits.json", "r") as f:
            kits: dict[str, list[str]] = json.load(f)
        if mac not in kits:
            return print(f"MAC address {mac} not found.")
        a = aliases.setdefault(mac, list())
        if alias in a:
            print(f"Alias {alias} already exists for MAC address {mac}.")
        else:
            a.append(alias)
            print(f"Alias {alias} added for MAC address {mac}.")
    elif action == "remove":
        # check if it exists
        if mac not in aliases:
            print(f"MAC address {mac} not found.")
        elif alias not in aliases[mac]:
            print(f"Alias {alias} not found for MAC address {mac}.")
        else:
            aliases[mac].remove(alias)
            print(f"Alias {alias} removed for MAC address {mac}.")
    else:
        print(f"Unknown action {action}.")
    # write aliases.json
    with open("aliases.json", "w") as f:
        json.dump(aliases, f)


def command_clearalias(args: list[str]):
    # check if aliases.json exists
    check_files()
    # read aliases.json
    with open("aliases.json", "r") as f:
        aliases: dict[str, list[str]] = json.load(f)
    mac = args[0]
    if mac not in aliases:
        print(f"MAC address {mac} not found.")
    else:
        del aliases[mac]
        print(f"Cleared all aliases for MAC address {mac}.")
    # write aliases.json
    with open("aliases.json", "w") as f:
        json.dump(aliases, f)


def command_list():
    # check if aliases.json exists
    check_files()
    # get macs from kits that are also in aliases
    with open("kits.json", "r") as f:
        kits: dict[str, list[str]] = json.load(f)
    with open("aliases.json", "r") as f:
        aliases: dict[str, list[str]] = json.load(f)
    real_macs = [mac for mac in kits if mac in aliases]

    if len(real_macs) == 0:
        print("No kits found.")
        return

    for mac in kits.items():
        if mac in aliases:
            print(f"{mac} -> {', '.join(aliases[mac])}")
        else:
            print(f"{mac}")


def check_if_valid_mac(mac):
    if len(mac) != 17:
        return False
    for i in range(0, 17, 3):
        if mac[i] != ":":
            return False
    return True


def get_mac_from_alias(alias):
    # check if aliases.json exists
    check_files()
    # read aliases.json
    with open("aliases.json", "r") as f:
        aliases: dict[str, list[str]] = json.load(f)
    for mac, aliases in aliases.items():
        if alias in aliases:
            return mac
    return None


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
