class LRC:
    @staticmethod
    def calculates(string):
        lrc = 127  # 0x7F
        for char in string:
            lrc ^= ord(char)
        return chr(lrc)


class StartOperation():
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

    def __init__(self):
        self.multilanguageFlag = StartOperation.Italian
        self.commandCode = None
        self.activateAsynchronousMessages = "0"  # Assuming Message::NO is '0'
        self.operationType = "0"
        self.transactionIdentifier = "00"
        self.finalAmountToDebit = "00000000"
        self.EMV = "1"
        self.RFU = "00"
        self.FIXED = "*"
        self.preauthCodeBCD = "0000"
        self.raw = None

    def add_lrc(self):
        lrc_char = LRC.calculates(self.raw)
        self.raw += lrc_char

    def get_string_to_send(self):
        self.raw = (
            STX
            + "terminalIdentifier"
            + self.multilanguageFlag  # Assuming this is a placeholder for an actual value
            + self.commandCode
            + self.activateAsynchronousMessages
            + self.operationType
            + self.transactionIdentifier
            + self.finalAmountToDebit
            + self.preauthCodeBCD
            + self.ETX
        )
        self.add_lrc()
