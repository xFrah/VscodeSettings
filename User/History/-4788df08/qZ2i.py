import network
import time


class WiFiManager:
    def __init__(self, ble):
        self.sta_if = network.WLAN(network.STA_IF)
        self.stop_retry = False
        self.ble = ble
        self.set_status(4)  # initial status (not yet initialized)

    def set_status(self, status):
        self.status = status
        self.ble.indicate_wifi_status(status)

    def connect_to_wifi(self, ssid, password, retries=5, retry_timeout=10, fallback=None):
        print("Connecting to Wi-Fi...")
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
