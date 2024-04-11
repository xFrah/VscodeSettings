# Description: Helper functions for the main.py file

import machine
import time


def reset_with_timestamp():
    """Write to file current time and a number then reset the device"""
    # check if file already exists
    try:
        with open("last_reset.txt", "r") as f:
            n = int(f.read().split(",")[1])
    except Exception as e:
        n = 0
    with open("last_reset.txt", "w") as f:
        f.write("{},{}".format(time.time(), n + 1))
    machine.reset()


def parse_reset_file():
    """Parse the reset file and return the time and reset cause"""
    try:
        with open("last_reset.txt", "r") as f:
            str_time, n = f.read().split(",")
            return float(str_time), int(n)
    except Exception as e:
        return None, None


def get_mac_address() -> str:
    """
    Get MAC address of the device

    Returns:
        str: MAC address
    """
    mac = "".join(["{:02X}".format(x) for x in machine.unique_id()])
    return ":".join([mac[i : i + 2] for i in range(0, len(mac), 2)])


def get_wifi_mac(sta_if):
    import ubinascii
    import network

    wlan_mac = sta_if.config("mac")
    my_mac_addr = ubinascii.hexlify(wlan_mac).decode()

    my_mac_addr = format_mac_addr(my_mac_addr)


def format_mac_addr(addr):

    mac_addr = addr
    mac_addr = mac_addr.upper()

    new_mac = ""

    for i in range(0, len(mac_addr), 2):
        # print(mac_addr[i] + mac_addr[i+1])

        if i == len(mac_addr) - 2:
            new_mac = new_mac + mac_addr[i] + mac_addr[i + 1]
        else:
            new_mac = new_mac + mac_addr[i] + mac_addr[i + 1] + ":"
    print("----------------------------------------")
    print("My MAC Address:" + new_mac)
    print("----------------------------------------")
    return new_mac
