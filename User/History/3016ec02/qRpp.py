from helpers import reset_with_timestamp, get_mac_address

from gc import collect
import machine

collect()
import time
import _thread
import json

collect()
from sec import security_object as sec_obj
import ble_client

print("[MAIN] Post-import ram check")


def main():
    print("[MAIN] Start of Main")
    collect()

    from web import WebServer
    from lib_mqtt import MQTTClient
    from wifi_manager import WMS

    WMS.connect_to_wifi()
    collect()

    webserver = WebServer()
    if WMS.is_connected():
        # TODO add timeout
        collect()
        res = webserver.update_database("https://e251cd87-8bc1-415d-84ab-938112ee7700.mock.pstmn.io/download", "00:11:22:33:44:55")
    collect()

    try:
        mqtt = MQTTClient(get_mac_address(), "5.196.23.212", port=1883)
        mqtt.connect()
    except Exception as e:
        mqtt = None

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
            "mac": get_mac_address(),
            "rfid": [(epc, rssi) for epc, rssi in tags.items()],
            # "timestamp": time.localtime(),
        }
        if sec_obj.authed == still_the_same:
            mqtt.publish("kits", f"{json.dumps(dictionary)}")
            print("[MAIN] Free memory:", gc.mem_free())

        time.sleep(0.3)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[MAIN] Error in main", e)
        reset_with_timestamp()
