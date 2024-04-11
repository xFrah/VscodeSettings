class LRC:
    @staticmethod
    def calculates(string):
        lrc = 127  # 0x7F
        for char in string:
            lrc ^= ord(char)
        return chr(lrc)


class StartOperation(Message):
    # Constants
    SOH = chr(1)
    STX = chr(2)
    ETX = chr(3)
    EOT = chr(4)
    ACK = chr(6)
    NAK = chr(21)

    # ... (other methods and variables)

    def add_lrc(self):
        lrc_char = LRC.calculates(self.raw)
        self.raw += lrc_char

    def get_string_to_send(self):
        self.raw = (
            self.STX +
            'terminalIdentifier' +  # Assuming this is a placeholder for an actual value
            self.multilanguageFlag +
            self.commandCode +
            self.activateAsynchronousMessages +
            self.operationType +
            self.transactionIdentifier +
            self.finalAmountToDebit +
            self.preauthCodeBCD +
            self.ETX
        )
        self.add_lrc()