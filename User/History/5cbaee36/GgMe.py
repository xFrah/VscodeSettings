from typing import Union
import serial
import time
from helpers import debug, get_configuration


class LRC:
    @staticmethod
    def calculates(string):
        lrc = 127  # 0x7F
        for char in string:
            lrc ^= ord(char)
        return chr(lrc)


def euros_to_padded_cents(euros):
    return f"{int(round(euros * 100)):08d}"


def receive_standard_pos(POS, timeout=10) -> Union[dict, None]:
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(0.5)
        buf = POS.read_all()
        if not buf:
            continue
        debug(str(buf))
        try:
            print("\n")
            msg = decode_packet(buf)
            for key, value in msg.items():
                print(" ", key, ":", value)
            return msg
            # print("Raw:", buf)
        except Exception as e:
            print(buf)
            pass


def activate_pos():
    """
    Sends the activate command to the POS. Raises an exception if the POS does not respond.
    """
    try:
        config = get_configuration()
        POS = serial.Serial(config["pos_port"], 115200, timeout=1)
        debug("POS opened")
        activate = StartOperation(config["terminal_id"])
        print(activate.get_string_to_send())
        for _ in range(3):
            debug("Sent activate")
            POS.write(activate.get_string_to_send().encode())
            packet = receive_standard_pos(POS, timeout=10)
            if packet and packet["Operation Code"] == "a":
                POS.close()
                return debug("Activate success")
    finally:
        try:
            POS.close()
        except:
            debug("POS not open")
    raise Exception("Activate failed")


class SendAmount:
    def __init__(self, amount) -> None:
        self.amount = amount
        self.raw = None

    def add_lrc(self):
        lrc_char = LRC.calculates(self.raw)
        self.raw += lrc_char

    def 


class StartOperation:
    # Constants
    SOH = chr(1)
    STX = chr(2)
    ETX = chr(3)
    EOT = chr(4)
    ACK = chr(6)
    NAK = chr(21)

    # Multilanguage Flags
    Italian = "0"
    German = "2"
    Spanish = "3"
    Portuguese = "4"
    French = "5"
    English = "6"

    # Command Codes
    Payment = "P"
    GetEMVTransactionData = "V"
    Reversal = "S"
    BankTotals = "Q"
    LocalTotals = "T"
    CloseSession = "C"
    DLL = "D"
    Realignment = "R"
    FirstDll = "L"
    ActivateEFTPOS = "a"
    DeactivateEFTPOS = "d"
    GetEFTPOSStatus = "s"
    RestartInstallation = "z"
    StartSoftwareMaintenance = "y"
    GetCardStatus = "G"
    ResetLog = "r"
    GetTerminalConfiguration = "c"
    GetAcquirerInformation = "e"
    GetAcquirerTotalAmounts = "l"
    GetGSMGPRSState = "g"
    RetrievingLastPaymentResult = "H"

    # Operation Types for "P" command
    PurchaseTransaction = "0"
    PreAuthorizationForFuelVendingMachines = "1"
    PreAuthorizationForOtherVendingMachines = "2"
    NotificationOfTransactionWithFinalAmountDebitTheCardForFuelVendingMachines = "3"
    NotificationOfTransactionWithFinalAmountDebitTheCardForOtherVendingMachines = "4"
    PurchaseTransactionWithMagstripeCardsWhenTheTracksAreReturnedAfterASlaveSession = (
        "5"
    )
    PurchaseTransactionToGetBINInClear = "9"

    # Codes for "S" command
    ReversalOfLastPurchaseOperation = "0"
    ReversalOfTheLastPreAuthorizationOperationForOtherVendingMachines = "2"
    ReversalOfLastPurchaseOperationWithoutCardInsertion = "3"

    def __init__(self, terminal_id, cmd=ActivateEFTPOS, amount: float = 0):
        self.multilanguageFlag = StartOperation.Italian
        self.terminal_id = terminal_id
        self.commandCode = cmd
        self.activateAsynchronousMessages = "1"  # Assuming Message::NO is '0'
        self.operationType = "0"
        self.transactionIdentifier = "00"
        self.finalAmountToDebit = euros_to_padded_cents(amount)
        self.preauthCodeBCD = "0000"
        self.raw = None

    def add_lrc(self):
        lrc_char = LRC.calculates(self.raw)
        self.raw += lrc_char

    def get_string_to_send(self):
        self.raw = (
            StartOperation.STX
            + self.terminal_id
            + self.multilanguageFlag  # Assuming this is a placeholder for an actual value
            + self.commandCode
            + self.activateAsynchronousMessages
            + self.operationType
            + self.transactionIdentifier
            + self.finalAmountToDebit
            + self.preauthCodeBCD
            + StartOperation.ETX
        )
        self.add_lrc()

        return self.raw


def decode_packet(packet_bytes):
    if packet_bytes[0:1] != b"\x02":
        raise Exception("Invalid packet")
    opcode = get_opcode(packet_bytes)
    return decode_dict[opcode](packet_bytes)


def get_opcode(packet_bytes):
    # the structure of all packets is the same in the first bytes
    # ("STX", 1),  # Start of Text
    # ("Terminal Identifier", 8),  # ASCII representation expected
    # ("Fixed Value", 1),  # ASCII representation expected
    # ("Operation Code", 1),  # ASCII representation expected

    return packet_bytes[10:11].decode("ascii", errors="ignore")


from packets import *

decode_dict = {
    "a": parse_ads,
    "d": parse_ads,
    "s": parse_ads,
    "F": parse_F,
    "E": parse_E,
    "I": parse_I,
}


if __name__ == "__main__":
    # config = get_configuration()
    config = {"terminal_id": "92541182"}
    start_operation = StartOperation(config["terminal_id"])
    print(start_operation.get_string_to_send())
    ser = serial.Serial("COM8", 115200, timeout=1)
    ser.write(start_operation.get_string_to_send().encode())
    while True:
        time.sleep(0.5)
        buf = ser.read_all()
        if not buf:
            continue
        try:
            print("\n")
            msg = decode_packet(buf)
            for key, value in msg.items():
                print(" ", key, ":", value)
            # print("Raw:", buf)
        except Exception as e:
            print(buf)
            pass
        # print(buf)
