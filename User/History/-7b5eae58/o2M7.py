import os
import json


def command_alias(args: list[str]):
    # check if aliases.json exists
    if not os.path.isfile("aliases.json"):
        with open("aliases.json", "w") as f:
            f.write("{}")
        print("Created aliases.json.")
    # read aliases.json
    with open("aliases.json", "r") as f:
        aliases: dict[str, list[str]] = json.load(f)
    action, mac, alias = args
    if action == "add":
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
    if not os.path.isfile("aliases.json"):
        with open("aliases.json", "w") as f:
            f.write("{}")
        return print("Created aliases.json.")
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
    if not os.path.isfile("aliases.json"):
        with open("aliases.json", "w") as f:
            f.write("{}")
        return print("Created aliases.json.")
    # read aliases.json
    with open("aliases.json", "r") as f:
        aliases: dict[str, list[str]] = json.load(f)
    for mac, alias in aliases.items():
        print(f"{mac} -> {', '.join(alias)}")
