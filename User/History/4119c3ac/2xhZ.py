from machine import Pin, UART
import _thread
import time

EXPIRY_DATE = 0
PRODUCT_ID = 1
MAX_PAYLOAD_SIZE = 700
TYPE_COMMAND = 0x00
TYPE_RESPONSE = 0x01
TYPE_NOTICE = 0x02
COMMAND_NOTICE = 0x22
COMMAND_ERROR = 0xFF
COMMAND_GET_TX_POWER = 0xB7
COMMAND_SET_TX_POWER = 0xB6
COMMAND_GET_REGION = 0x08
COMMAND_GET_MODEM_PARAMETERS = 0xF1
COMMAND_SET_MODEM_PARAMETERS = 0xF0
COMMAND_SET_FREQUENCY_HOPPING = 0xAD

RESPONSE_CODES = {
    0x17: "Invalid command",
    0x20: "Frequency hopping time out. All channel are occupied",
    0x15: "Inventory fail. No tag response or CRC error",
    0x16: "Access fail. May caused by password error",
    0x09: "Read fail. No tag response or CRC error",
    0x10: "Write fail. No tag response or CRC error",
    0x13: "Lock or Kill fail. No tag response or CRC error",
}

REGION_CODES = {
    0x01: "China 900MHZ",
    0x04: "China 800MHZ",
    0x02: "USA",
    0x03: "Europe",
    0x05: "Korea",
}


def validate_packet(packet: bytes) -> bool:
    """
    Validates packet by checking if checksum is correct
    """
    checksum_byte = packet[-2]
    checksum = sum(packet[1:-2])
    checksum = checksum & 0xFF
    flag = checksum_byte == checksum
    if not flag:
        print(f"Checksum failed: {checksum_byte} != {checksum}")
    return flag


def get_all_headers(arr: bytes) -> list[int]:
    """
    Returns all indexes of headers and ends
    """
    return [i for i, x in enumerate(arr) if x == 0xBB]


class UHF:
    """
    Initializes rfid module and opens uart connection
    """

    def __init__(self) -> None:

        self.packets = []
        # self.ser = serial.Serial("COM5", 115200, timeout=0.1)
        self.ser = UART(1, baudrate=115200, tx=14, rx=34)
        self.ser.init()
        # self.ser.init(115200, bits=8, parity=None, stop=1)

        # threading.Thread(target=self.receiver, daemon=True).start()
        _thread.start_new_thread(self.receiver, ())
        time.sleep(1.5)
        self.setup()

    def setup(self) -> None:
        """
        Sets up the rfid module.
        """
        self.set_tx_power(2600)
        self.get_tx_power()
        self.get_region()
        self.set_modem_params(0x02, 0x06, 0x00B0, wait=0.7)
        self.get_modem_params(wait=1)
        self.set_frequency_hopping(wait=0.7)

    def receiver(self):
        """
        Asynchronously receives data from the RFID module through the UART connection.
        Parses the received data into packets and validates them before adding them to the packets list.
        """

        data_buffer = bytearray()
        while True:
            time.sleep(0.01)
            res = self.ser.read()
            # print(res)
            if res == b"" or res is None:
                time.sleep(0.1)
                continue

            # print(f"Lenght+: {len(res)}, total: {len(data_buffer) + len(res)}")

            data_buffer += res

            first_header = data_buffer.find(b"\xBB")
            counter = 0
            while first_header != -1 and len(data_buffer) >= first_header + 5:
                # del data_buffer[:first_header]
                # bytearray doesn't support item deletion, just slicing
                data_buffer[:first_header] = b""
                self.os_buffer_warning()
                pl = (data_buffer[3] << 8) + data_buffer[4]  # payload length
                if pl > MAX_PAYLOAD_SIZE:
                    # del data_buffer[0]
                    data_buffer[:1] = b""
                    print(f"Packet exceeds max payload size: {pl} > {MAX_PAYLOAD_SIZE}.")
                    break
                end = pl + 6  # there are 6 bytes before payload
                if len(data_buffer) < end:
                    # print("Not enough data 2")
                    break
                packet = data_buffer[: end + 1]
                if validate_packet(packet):
                    self.packets.append(Packet(packet))
                    # del data_buffer[: end + 1]
                    data_buffer[: end + 1] = b""
                    # print(f"Valid packet found")
                    counter += 1
                else:
                    data_buffer = data_buffer[1:]
                    # print(f"Failed to validate packet")
                # print(f"Found {counter} packets in {len(res)} bytes")
                first_header = data_buffer.find(b"\xBB")
        print("Done")

    def os_buffer_warning(self):
        """
        Prints a warning if the buffer is too full. It should be emptied as soon as possible.
        """
        # if self.ser.in_waiting > 1000:
        #     print(f"Buffer warning: {self.ser.in_waiting} bytes in buffer")

        # from serial to uart
        n = self.ser.any()
        if n > 1000:
            print(f"Buffer warning: {n} bytes in buffer")

    def _write(self, arr: bytes, wait):
        """
        Writes bytes to the serial stream
        """
        time.sleep(wait)
        self.ser.write(arr)

    def set_frequency_hopping(self, wait=0.2) -> None:
        """
        Gets the frequency hopping of the rfid module
        """
        self._write(b"\xBB\x00\xAD\x00\x01\xFF\xAD\x7E", wait=wait)

    def get_tx_power(self, wait=0.2) -> int:
        """
        Gets the tx power of the rfid module in dBm. Ex. 2000 is 20 dBm.
        """
        self._write(b"\xBB\x00\xB7\x00\x00\xB7\x7E", wait=wait)

    def set_tx_power(self, power: int, wait=0.2) -> None:
        """
        Sets the tx power of the rfid module
        """
        # power must be encoded in 2 bytes
        # convert integer power into msb and lsb
        msb = power >> 8
        lsb = power & 0xFF
        checksum = sum(b"\x00\xB6\x00\x02" + bytes([msb, lsb])) & 0xFF
        self._write(b"\xBB\x00\xB6\x00\x02" + bytes([msb, lsb, checksum]) + b"\x7E", wait=wait)

    def set_modem_params(self, mix_gain: int, if_gain: int, threshold: int, wait=0.2) -> None:
        """
        Sets the modem parameters of the rfid module
        """
        # power must be encoded in 2 bytes
        # convert integer power into msb and lsb
        checksum = sum(b"\x00\xF0\x00\x04" + bytes([mix_gain, if_gain, threshold >> 8, threshold & 0xFF])) & 0xFF
        self._write(b"\xBB\x00\xF0\x00\x04" + bytes([mix_gain, if_gain, threshold >> 8, threshold & 0xFF, checksum]) + b"\x7E", wait=wait)

    def get_modem_params(self, wait=0.2):
        """
        Gets the modem parameters of the rfid module
        """
        self._write(b"\xBB\x00\xF1\x00\x00\xF1\x7E", wait=wait)

    def get_region(self, wait=0.2) -> int:
        """
        Gets the region of the rfid module
        """
        self._write(b"\xBB\x00\x08\x00\x00\x08\x7E", wait=wait)

    def deep_scan(self, nfc, timeout=30) -> dict[str, int]:
        """
        Scans for rfid tags nearby and returns a list of tags with their average rssi.
        """
        permasaw = {}
        self.scan()
        start = time.time()
        packets_number = 0
        step_n = 0
        max_n = 0
        while time.time() - start < timeout and nfc.closed:
            tags, other = self.get_data()
            # print("\nPackets:", len(tags) + len(other))
            packets_number += len(tags) + len(other)
            for tag in tags:
                hashable = "-".join([f"{byte:08b}" for byte in tag.epc])
                if hashable not in permasaw:
                    permasaw[hashable] = [tag.rssi]
                else:
                    permasaw[hashable].append(tag.rssi)
            if len(permasaw) > max_n:
                max_n = len(permasaw)
                step_n = packets_number + 0
                print(f"Found {max_n} tags")
                print(permasaw)
                # self.get_modem_params()
            if other:
                for packet in other:
                    if packet._data["msg"] != "Inventory fail. No tag response or CRC error":
                        print(packet)
        self.stop_scan()
        print(f"Found {len(permasaw)} tags in {packets_number}({step_n}) packets.")
        averages = {}
        for tag, values in permasaw.items():
            values = values[-10:] if len(values) > 11 else values
            averages[tag] = sum(values) / len(values)
        return averages

    def get_data(self, start=None):
        """
        Iterates through the packet buffer and returns a list of tags and other packets.
        """
        tags: list[Tag] = []
        other_packets: list[Packet] = []

        def remove_packet(packet: Packet):  # TODO acquire lock here
            try:
                self.packets.remove(packet)
            except ValueError:
                print("[RFID] Packet already removed")

        # TODO maybe acquire lock here
        for packet in self.packets.copy():
            packet._decode()
            if start is not None and packet.timestamp < start:
                remove_packet(packet)
                print("[RFID] Packet expired")
            elif packet.packet_type == "tag_notice":
                remove_packet(packet)
                tags.append(Tag(packet["epc"], packet["rssi"]))
            else:
                remove_packet(packet)
                other_packets.append(packet)
        return tags, other_packets

    def scan(self, wait=0.1) -> list[Tag]:
        """
        Scan for rfid tags nearby and returns a list of tags. It stops by itself after a while, it is advised to use stop_scan().
        """
        self._write(b"\xBB\x00\x27\x00\x03\x22\x27\x10\x83\x7E", wait=wait)

    def stop_scan(self, wait=0.1) -> None:
        """
        Stops scanning for rfid tags
        """
        self._write(b"\xBB\x00\x28\x00\x00\x28\x7E", wait=wait)


class Packet:
    """
    Automatically decodes packet bytes into a readable format.
    The packet doesn't get decoded until an attribute is accessed, as to minimize latency in UART readings.

    Args:
    - raw (bytes): The raw bytes from the serial stream.
    """

    def __init__(self, raw: bytes) -> None:
        self.raw: bytes = raw
        """Raw packet data"""
        self.packet_type: str = None
        """Type of packet, either 'tag_notice' or 'error'"""
        self._decoded: bool = False
        """Private variable to check if packet has been decoded"""
        self.timestamp = time.time()

    def _decode(self) -> None:
        """
        Decodes packet data into a private dictionary
        """
        self._data: dict[str, any] = {}
        packet_type = self.raw[1]
        command = self.raw[2]
        if packet_type == TYPE_NOTICE and command == COMMAND_NOTICE:
            # rssi is byte 5, pc is bytes 6 to 7, epc is bytes 8 to 19, crc is bytes 20 to 21
            self.packet_type = "tag_notice"
            self._data["rssi"] = int(-0xFF + self.raw[5])
            self._data["pc"] = self.raw[6:8]
            self._data["epc"] = self.raw[8:20]
            self._data["crc"] = self.raw[20:22]
        elif packet_type == TYPE_RESPONSE:
            self.packet_type = "response"
            if command == COMMAND_GET_TX_POWER:
                self._data["tx_power"] = (self.raw[5] << 8) + self.raw[6]
                self._data["msg"] = "TX power is " + str(self._data["tx_power"]) + " dBm."
            elif command == COMMAND_SET_TX_POWER:
                self._data["msg"] = "TX power has been successfully changed."
            elif command == COMMAND_GET_REGION:
                self._data["region"] = self.raw[5]
                region_name = REGION_CODES[self._data["region"]] if self._data["region"] in REGION_CODES else str(self._data["region"])
                self._data["msg"] = f"Region is {region_name}."
            elif command == COMMAND_ERROR:
                self.packet_type = "error"
                self._data["msg"] = RESPONSE_CODES[self.raw[5]] if self.raw[5] in RESPONSE_CODES else self.raw[5]
            elif command == COMMAND_GET_MODEM_PARAMETERS:
                self._data["mix_gain"] = self.raw[5]
                self._data["if_gain"] = self.raw[6]
                # threshold is bytes 7 to 8
                self._data["threshold"] = (self.raw[7] << 8) + self.raw[8]
                self._data["msg"] = "Modem parameters received."
            elif command == COMMAND_SET_MODEM_PARAMETERS:
                self._data["msg"] = "Modem parameters have been successfully changed."
            elif command == COMMAND_SET_FREQUENCY_HOPPING:
                self._data["msg"] = "Frequency hopping has been successfully enabled."
            else:
                self._data["msg"] = "Unknown response"
        else:
            self.packet_type = "unknown"
            self._data["msg"] = "Unknown packet type"
        self.decoded = True

    def __dict__(self):
        if not self._decoded:
            self._decode()
        return self._data

    def __getitem__(self, key):
        if not self._decoded:
            self._decode()
        return self._data[key]

    # string representation
    def __str__(self):
        if not self._decoded:
            self._decode()
        # check if "msg" in data
        if "msg" in self._data:
            t_dict = self._data.copy()
            del t_dict["msg"]
            return f"[{self.packet_type.upper()}] " + self._data["msg"] + f": {t_dict}"
        return self._data
