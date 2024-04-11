from helpers import reset_with_timestamp, get_mac_address, get_ble_mac

print(f"blemac {get_ble_mac()}")

from gc import collect
import ntptime
import machine
import utime

collect()
import time
import _thread
import json

collect()
from sec import security_object as sec_obj
from updater import check_for_updates

print("[MAIN] Post-import ram check")


def set_time():
    ntptime.host = "pool.ntp.org"  # You can use other NTP servers if needed
    ntptime.settime()  # Set the time
    (year, month, mday, hour, minute, second, weekday, yearday) = utime.localtime()
    # Adjusting for Rome's time zone (CET/CEST)
    hour = (hour + 2) % 24  # Adjusting for CET (UTC+2)
    # Check for daylight saving time (last Sunday in March to last Sunday in October)
    if (3 < month < 10) or (month == 3 and mday - weekday > 24) or (month == 10 and mday - weekday <= 24):
        hour = (hour + 1) % 24  # Adjusting for CEST (UTC+3)
    # Set the adjusted time
    rtc = machine.RTC()
    rtc.datetime((year, month, mday, weekday, hour, minute, second, 0))


def main():
    print("[MAIN] Start of Main")
    collect()

    from web import WebServer
    from lib_mqtt import MQTTClient
    from wifi_manager import WMS

    WMS.connect_to_wifi()
    collect()

    set_time()

    check_for_updates()

    try:
        mqtt = MQTTClient(get_ble_mac(), "5.196.23.212", port=1883)
        mqtt.connect()
    except Exception as e:
        mqtt = None

    import ble_client
    from rfid import UHF

    rfid = UHF()
    print("[MAIN] Setup complete")

    while True:
        collect()
        still_the_same = sec_obj.authed

        if sec_obj.authed:
            timeout = 4
            careless = True
        else:
            timeout = 15
            careless = False

        tags: dict[int, str] = rfid.deep_scan(timeout=timeout, careless=careless)
        dictionary = {
            "status": "closed" if sec_obj.closed else "open",
            "authenticated": sec_obj.authed,
            "mac": get_ble_mac(),
            "rfid": [(epc, rssi) for epc, rssi in tags.items()],
            # "timestamp": time.localtime(),
        }
        if sec_obj.authed == still_the_same and mqtt is not None:
            mqtt.publish("kits", f"{json.dumps(dictionary)}")
            print("[MAIN] Free memory:", gc.mem_free())

        time.sleep(0.3)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[MAIN] Error in main", e)
        reset_with_timestamp()
