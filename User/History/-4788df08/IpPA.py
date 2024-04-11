import network
import time
import _thread

wifi_credentials_lock = _thread.allocate_lock()


def get_wifi_credentials() -> (str, str) or (None, None):  # TODO move in wifi_manager.py
    """
    Get wifi credentials from file

    Returns:
        (str, str) or (None, None): SSID and password of wifi network
    """

    with wifi_credentials_lock:
        try:
            wifi_dict = json.load(open("wifi.json", "r"))
        except Exception as e:
            print(e)
            return None, None
        # check if malformed
        if "ssid" not in wifi_dict.keys() or "password" not in wifi_dict.keys():
            return None, None
        return wifi_dict["ssid"], wifi_dict["password"]


def write_wifi_credentials(ssid: str, password: str) -> bool:  # TODO move in wifi_manager.py
    """
    Write wifi credentials to file

    Args:
        ssid (str): SSID of wifi network
        password (str): Password of wifi network

    Returns:
        bool: True if successful, False otherwise
    """
    with wifi_credentials_lock:
        try:
            json.dump({"ssid": ssid, "password": password}, open("wifi.json", "w"))
        except Exception as e:
            print(e)
            return False
    return True


class WiFiManager:
    def __init__(self, ble=None):
        self.sta_if = network.WLAN(network.STA_IF)
        self.stop_retry = False
        self.ble = ble
        self.set_status(3)  # initial status (not yet initialized)

    def set_status(self, status):  # TODO this is complete dogshit, the status should be set by the BLE server by polling
        if self.ble is None:
            return
        # status: 0 = not connected, 1 = connected, 2 = trying to connect, 3 = not initialized
        print("[WIFI] Setting Wi-Fi status to", status)
        self.ble.indicate_wifi_status(status)

    def connect_to_wifi(self, retries=5, retry_timeout=10, fallback=None):
        ssid, password = get_wifi_credentials()
        print("[WIFI] Connecting to Wi-Fi...")
        self.set_status(2)  # status: trying to connect
        if self.sta_if.isconnected():  # if already connected, do nothing
            self.set_status(1)  # status: connected
            return True

        self.sta_if.active(True)  # activate the interface
        self.sta_if.connect(ssid, password)  # start connection attempt
        self.stop_retry = False  # reset the stop flag

        attempt = 0
        while not self.sta_if.isconnected() and not self.stop_retry:  # wait for it to connect or stop flag to be set
            attempt += 1
            if attempt > retries:
                print("Failed to connect to Wi-Fi after", retries, "attempts")
                self.set_status(0)  # status: not connected
                if fallback is not None:
                    fallback()  # call the fallback function if it's provided
                return False

            print("Connection attempt", attempt, "failed, retrying in", retry_timeout, "seconds...")
            time.sleep(retry_timeout)  # wait before retrying

        if self.stop_retry:
            print("Connection attempts stopped.")
            self.set_status(0)  # status: not connected
            return False

        print("Connected to Wi-Fi")
        self.set_status(1)  # status: connected
        return True

    def stop_connect(self):
        self.stop_retry = True

    def is_connected(self):
        return self.sta_if.isconnected()

    def get_network_config(self):
        return self.sta_if.ifconfig()

    def get_status(self):
        return self.status


WMS = WiFiManager()
