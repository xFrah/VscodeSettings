import bluetooth
import random
import struct
import network
import time
import _thread
import machine
from ble_advertising import advertising_payload
from wifi_manager import WMS, get_wifi_credentials, write_wifi_credentials
from sec import security_object as sec_obj

from micropython import const

_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)
_IRQ_GATTS_INDICATE_DONE = const(20)
_IRQ_MTU_EXCHANGED = const(21)

_FLAG_READ = const(0x0002)
_FLAG_NOTIFY = const(0x0010)
_FLAG_WRITE = const(0x0008)
_FLAG_INDICATE = const(0x0020)

# org.bluetooth.service.environmental_sensing
_COMMANDS_SERVICE_UUID = bluetooth.UUID(0x181A)
_WIFI_SERVICE_UUID = bluetooth.UUID(0x181B)
# org.bluetooth.characteristic.temperature

_COMMANDS_INPUT = (
    bluetooth.UUID("6E400002-B5A3-F393-E0A9-E50E24DCCA9E"),
    _FLAG_WRITE | _FLAG_INDICATE | _FLAG_READ,
)
_COMMANDS_SERVICE = (
    _COMMANDS_SERVICE_UUID,
    (_COMMANDS_INPUT,),
)

_WIFI_STATUS = (
    bluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E"),
    _FLAG_NOTIFY | _FLAG_READ | _FLAG_INDICATE,
)

_WIFI_SERVICE = (
    _WIFI_SERVICE_UUID,
    (_WIFI_STATUS,),
)


# org.bluetooth.characteristic.gap.appearance.xml
_ADV_APPEARANCE_GENERIC_THERMOMETER = const(0x0D80)


class BLEPeripheral:
    def __init__(self, name="First-Aid-Kit"):
        self._ble = bluetooth.BLE()
        self.name = name
        self.commands = {}
        self.start()

    def confirm_result(self, result: bool, value_handle: int, value: bytearray) -> None:
        """
        Confirm result of a command by writing to the characteristic

        Args:
            result (bool): True if command was successful, False otherwise
            value_handle (int): handle of the characteristic
            value (bytearray): value of the characteristic
        """
        cmd: bytearray = value + b"=1" if result else b"=0"
        print("[BLE] Confirming result", cmd)
        self._ble.gatts_write(value_handle, cmd)
        # indicate
        for conn_handle in self._connections:
            self._ble.gatts_indicate(conn_handle, value_handle)

    def start(self):
        self._ble.active(True)
        self._ble.irq(self._irq)
        ((self._handle_commands_input,), (self._handle_wifi_status,)) = self._ble.gatts_register_services(
            (_COMMANDS_SERVICE, _WIFI_SERVICE)
        )
        self._ble.gatts_set_buffer(self._handle_commands_input, 256)
        # set mtu with config
        self._ble.config(mtu=256)
        self._connections = set()
        self._payload = advertising_payload(
            name=self.name, services=[_COMMANDS_SERVICE_UUID, _WIFI_SERVICE_UUID], appearance=_ADV_APPEARANCE_GENERIC_THERMOMETER
        )
        self._advertise()
        print("[BLE] Started advertising.")
        # start command thread
        _thread.start_new_thread(self.cmd_thread, ())

    def cmd_thread(self):
        print("[BLE] Command thread started.")
        while True:
            try:
                time.sleep(0.1)
                if self._connections:
                    return self.notify_wifi_status(WMS.update_status())
                if len(self.commands) > 0:
                    cmds = list(self.commands.items())
                    self.commands.clear()
                    for command, (args, whole, value_handle) in cmds:
                        try:
                            if command == b"wifi_save":
                                ssid, password = args[0], args[1]
                                print("[MAIN] Received wifi credentials", ssid, password)
                                # res: bool = write_wifi_credentials(ssid, password)
                                self.confirm_result(True, value_handle, whole)
                                print(f"[MAIN] {'Saved' if res else 'Failed to save'} wifi credentials, restarting...")
                                # time.sleep(5)
                                # TODO acquire all locks, because this is in another thread
                                # machine.reset()

                            elif command == b"magnetometer_calibrate":
                                print("[MAIN] Received magnetometer calibration command")
                                sec_obj.need_calibration = True
                                self.confirm_result(True, value_handle, whole)

                            elif command == b"item_reg":
                                item_type = args[0]
                                tags = self._rfid.scan()
                                closest = max(tags, key=lambda x: x.rssi)
                                print("[MAIN] Received item registration command", item_type, closest)
                                # TODO save item through HTTP request
                                self.confirm_result(True, value_handle, whole)

                            elif command == b"mqtt_save":
                                print("[MAIN] Received mqtt credentials")
                                # TODO save mqtt credentials
                                # TODO restart

                            elif command == b"restart":
                                print("[MAIN] Received restart command")
                                # TODO acquire all locks, because this is in another thread
                                self.confirm_result(True, value_handle, value)
                                machine.reset()
                            else:
                                print("[MAIN] Received unknown command", command)
                        except Exception as e:
                            print("[MAIN] Error while executing command", command, e)
                        time.sleep(0.1)
            except Exception as e:
                print("[BLE] Error in command thread:", e)

    def stop(self):
        self._ble.active(False)

    def notify_wifi_status(self, status: int) -> None:
        """
        Notify connected centrals of wifi status

        Args:
            status (int): wifi status
        """
        # print("[BLE] Indicating wifi status", status)
        self._ble.gatts_write(self._handle_wifi_status, str(status).encode())
        for conn_handle in self._connections:
            self._ble.gatts_notify(conn_handle, self._handle_wifi_status)

    def _irq(self, event, data):
        # Track connections so we can send notifications.
        if event == _IRQ_CENTRAL_CONNECT:
            conn_handle, addr_type, addr = data
            self._connections.add(conn_handle)
            print("[BLE] New connection", conn_handle, addr_type, addr)
            self._advertise()
            # ask mtu
            self._ble.gattc_exchange_mtu(conn_handle)
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn_handle, addr_type, addr = data
            self._connections.remove(conn_handle)
            print("[BLE] Disconnected", conn_handle, addr_type, addr)
            # Start advertising again to allow a new connection.
            self._advertise()
        elif event == _IRQ_GATTS_INDICATE_DONE:
            conn_handle, value_handle, status = data
            # print("[BLE] Indication sent")
        elif event == _IRQ_MTU_EXCHANGED:
            conn_handle, mtu = data
            print("[BLE] MTU exchanged", conn_handle, mtu)
        elif event == _IRQ_GATTS_WRITE:
            conn_handle, value_handle = data
            value: bytearray = self._ble.gatts_read(value_handle)
            print("[BLE] Write", value, conn_handle, value_handle, (self._handle_commands_input))
            # print mtu
            print(self._ble.config("mtu"))

            if value_handle == self._handle_commands_input:
                # message should be of form command:wifi_save+SSID+PASS, parse it
                if value.startswith(b"command:"):
                    try:
                        parsed: List[bytearray] = value.split(b"+", 2)
                        command: bytearray = parsed[0][8:]
                        self.commands[command] = (parsed[1:], value, value_handle)
                        print("[BLE] Received command", command, parsed[1:])
                    except Exception as e:
                        print(f"[BLE] Error parsing command: \n{e}")
                else:
                    print('[BLE] Commands must start with "command:"', value)

    # def set_temperature(self, temp_deg_c, notify=False, indicate=False):
    # Data is sint16 in degrees Celsius with a resolution of 0.01 degrees Celsius.
    # Write the local value, ready for a central to read.
    # self._ble.gatts_write(self._handle, struct.pack("<h", int(temp_deg_c * 100)))
    # if notify or indicate:
    #    for conn_handle in self._connections:
    #        if notify:
    # Notify connected centrals.
    #            self._ble.gatts_notify(conn_handle, self._handle)
    #        if indicate:
    # Indicate connected centrals.
    #            self._ble.gatts_indicate(conn_handle, self._handle)

    def _advertise(self, interval_us=500000):
        self._ble.gap_advertise(interval_us, adv_data=self._payload)


ble = BLEPeripheral()
