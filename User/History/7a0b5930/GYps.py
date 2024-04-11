from datetime import datetime
import json
import serial


def padded_cents_to_euros(padded_cents_str):
    """
    Converts a padded cents string back to euros.

    Args:
    padded_cents_str (str): A string representing the cents padded to 8 digits.

    Returns:
    float: The euro amount corresponding to the cents string.
    """
    # Convert the string to an integer to get the cents
    cents = int(padded_cents_str)

    # Divide by 100 to convert cents to euros
    return cents / 100


def parse_receipt_string(input_string):
    """
    Parses the provided string into 5 rows of 24 characters each.

    Args:
    input_string (str): The string to be parsed.

    Returns:
    list: A list of 5 strings, each representing a row.
    """
    # Splitting the string into 5 rows of 24 characters each
    rows = [input_string[i : i + 24] for i in range(0, len(input_string), 24)]

    # Ensuring there are exactly 5 rows, even if some are empty
    rows += [""] * (5 - len(rows))

    return rows


class Printer:
    def __init__(self, port, options=None):
        if options is None:
            options = {}
        for key, value in options.items():
            setattr(self, key, value)

        self.device = serial.Serial(port, 115200)
        self.init_printer()
        self.get_printer_full_status()
        self.dump_printer_full_status()

    def buffer(self, string):
        self.device.write(string.encode("ascii"))

    def buffer_span_row(self, total_length, strings):
        """
        totalLength: la lunghezza totale in caratteri di una riga e dato un array di stringhe,
        il metodo stampa in modo tale da spalmare tutte le stringhe lungo tutti i totalLength caratteri a disposizione.
        """
        if isinstance(strings, str):
            strings = [strings]
        gaps = len(strings) - 1
        if gaps > 0:
            length = sum(len(s) for s in strings) + gaps
            space = (total_length - length) // gaps
            extra_space = (total_length - length) % gaps
            for i, string in enumerate(strings[:-1]):
                self.device.write(string)
                self.device.write(" " * space)
                if i < extra_space:
                    self.device.write(" ")
            if strings[-1]:
                self.device.write(strings[-1])
        self.line_feed()

    def buffer_left_right(self, string_left, string_right, total_length):
        string = (
            string_left
            + " " * (total_length - len(string_left) - len(string_right))
            + string_right
        )
        self.device.write(string)

    def get_printer_full_status(self):
        self.device.write(bytearray([16, 4, 20]))
        raw = self.device.read(6)
        if len(raw) == 6:
            paper_status = ord(raw[2:3])
            self.paper_missing = bool(paper_status & 1)
            self.almost_paper_missing = bool(paper_status & 4)
            self.ticket_out = bool(paper_status & 32)
            self.notch_not_found = bool(paper_status & 128)
            recoverable_error_status = ord(raw[4:5])
            self.printer_head_temp_failure = bool(recoverable_error_status & 1)
            self.com_port_error = bool(recoverable_error_status & 2)
            self.power_supply_error = bool(recoverable_error_status & 8)
            self.command_unknown = bool(recoverable_error_status & 32)
            self.paper_jam = bool(recoverable_error_status & 64)
            unrecoverable_error_status = ord(raw[5:6])
            self.cutter_error = bool(unrecoverable_error_status & 1)
            self.ram_error = bool(unrecoverable_error_status & 4)
            self.eprom_error = bool(unrecoverable_error_status & 8)
        else:
            raise Exception("Printer is not communicating")

    def can_print(self):
        return all(
            not getattr(self, attr)
            for attr in [
                "paper_missing",
                "printer_head_temp_failure",
                "power_supply_error",
                "paper_jam",
                "cutter_error",
                "ram_error",
                "eprom_error",
            ]
        )

    def init_printer(self):
        self.device.write(bytearray([27, 64]))  # Initialize printer

    def line_feed(self):
        self.device.write(bytearray([10]))

    def print_raster_image(self, rows, black="1"):
        num_rows = len(rows)
        num_cols = len(rows[0]) if rows else 0

        # Check if rows is a properly formed matrix
        if any(len(row) != num_cols for row in rows):
            raise Exception(
                "Malformed Matrix Image (all rows don't have the same amount of elements)"
            )
        if num_cols % 8 != 0:
            raise Exception("Malformed Matrix Image (columns must be a multiple of 8)")

        xH = num_cols // 8 // 256
        xL = num_cols // 8 % 256
        yH = num_rows // 256
        yL = num_rows % 256
        ord_values = []
        for row in rows:
            ord_val = 0
            count_bit = 7
            for char in row:
                if char == black:
                    ord_val += 2**count_bit
                count_bit -= 1
                if count_bit < 0:
                    ord_values.append(chr(ord_val))
                    ord_val = 0
                    count_bit = 7
        string = "".join(ord_values)
        self.device.write(
            chr(29)
            + chr(118)
            + chr(48)
            + chr(0)
            + chr(xL)
            + chr(xH)
            + chr(yL)
            + chr(yH)
            + string
        )

    def set_cpi(self, n=0):
        """
        n = {0,48} => Font A = 16cpi, Font B = 23cpi
        n = {1,49} => Font A = 23cpi, Font B = 30cpi
        """
        self.device.write(bytearray([27, 193, int(n)]))

    def set_printing_mode(self, n=0):
        self.device.write(bytearray([27, 33, int(n)]))

    def right_char_spacing(self, n=64):
        self.device.write(bytearray([27, 32, int(n)]))

    def set_chars_upside_down(self, n=0):
        self.device.write(bytearray([27, 123, int(n)]))

    def line_spacing(self, n=64):
        self.device.write(bytearray([27, 51, int(n)]))

    def set_double_printing(self, n=0):
        self.device.write(bytearray([27, 71, int(n)]))

    def set_char_size(self, n=0):
        """
        n = 0 =>        normale
        """
        self.device.write(bytearray([29, 33, n]))

    def set_expanded_mode(self, n=0):
        """
        n = 0 => disabled
        n = 1 => enabled
        """
        self.device.write(bytearray([27, 69, n]))

    def total_cut(self):
        self.device.write(bytearray([27, 105]))

    def form_feed(self):
        self.device.write(bytearray([12]))

    def feed(self):
        self.device.write(bytearray([27, 100, 0]))

    def left_alignment(self):
        self.device.write(bytearray([27, 97, 0]))

    def center_alignment(self):
        self.device.write(bytearray([27, 97, 1]))

    def right_alignment(self):
        self.device.write(bytearray([27, 97, 2]))

    def set_left_margin(self, nL, nH):
        self.device.write(bytearray([29, 76, nL, nH]))

    def set_bold(self, active=0):
        self.device.write(bytearray([27, 71, active]))

    def set_character_size(self, width=0, height=0):
        self.device.write(bytearray([29, 33, width + height]))

    def print_lines_array(self, lines_array):
        for line in lines_array:
            if "cmd" in line and line["cmd"]:
                params = line.get("params", {})
                if isinstance(params, list):
                    getattr(self, line["cmd"])(*params)
                else:
                    getattr(self, line["cmd"])(**params)

    def dump_printer_full_status(self):
        print(f'Carta Mancante: {"YES" if self.paper_missing else "NO"}')
        print(f'Carta quasi Mancante: {"YES" if self.almost_paper_missing else "NO"}')
        print(f'Carta fuori: {"YES" if self.ticket_out else "NO"}')
        print(f'Notch non trovato: {"YES" if self.notch_not_found else "NO"}')
        print(f'Errore testina: {"YES" if self.printer_head_temp_failure else "NO"}')
        print(f'Errore porta seriale: {"YES" if self.com_port_error else "NO"}')
        print(f'Errore alimentazione: {"YES" if self.power_supply_error else "NO"}')
        print(f'Comando sconosciuto: {"YES" if self.command_unknown else "NO"}')
        print(f'Inceppamento carta: {"YES" if self.paper_jam else "NO"}')
        print(f'Errore cutter: {"YES" if self.cutter_error else "NO"}')
        print(f'Errore RAM: {"YES" if self.ram_error else "NO"}')
        print(f'Errore EPROM: {"YES" if self.eprom_error else "NO"}')

    def dump_printer_status_dict(self):
        return {
            "paper_missing": self.paper_missing,
            "almost_paper_missing": self.almost_paper_missing,
            "ticket_out": self.ticket_out,
            "notch_not_found": self.notch_not_found,
            "printer_head_temp_failure": self.printer_head_temp_failure,
            "com_port_error": self.com_port_error,
            "power_supply_error": self.power_supply_error,
            "command_unknown": self.command_unknown,
            "paper_jam": self.paper_jam,
            "cutter_error": self.cutter_error,
            "ram_error": self.ram_error,
            "eprom_error": self.eprom_error,
        }

    def to_array(self):
        return {k: v for k, v in self.__dict__.items() if k != "device"}

    def to_json(self):
        import json

        return json.dumps(self.to_array())

    def close(self):
        try:
            self.device.close()
        except Exception as e:
            raise Exception(str(e))

    def set_barcode_width(self, n=3):
        self.device.write(bytearray([29, 119, n]))

    def print_barcode(self, m=8, code="123456"):
        self.center_alignment()
        self.device.write(
            bytearray([29, 107, m]) + code.encode("ascii") + bytearray([0])
        )


if __name__ == "__main__":
    # get datetime as this: 08/06/17
    # text_lines_json = '[{"cmd":"init_printer"},{"cmd":"set_cpi","params":{"n":49}},{"cmd":"set_printing_mode","params":{"n":1}},{"cmd":"right_char_spacing","params":[0]},{"cmd":"set_chars_upside_down","params":[1]},{"cmd":"set_left_margin","params":{"nL":10,"nH":0}},{"cmd":"center_alignment"},{"cmd":"line_spacing","params":[24]},{"cmd":"buffer","params":["Valido fino a "]},{"cmd":"set_double_printing","params":[1]},{"cmd":"buffer","params":["giorno scadenza"]},{"cmd":"set_double_printing","params":[0]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Emesso il "]},{"cmd":"set_double_printing","params":[1]},{"cmd":"buffer","params":["08/06/17"]},{"cmd":"set_double_printing","params":[0]},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"set_left_margin","params":{"nL":5,"nH":0}},{"cmd":"buffer","params":["Esente IVA Art.10 C.1 n.22 DPR 633/72 SM"]},{"cmd":"set_left_margin","params":{"nL":10,"nH":0}},{"cmd":"line_feed"},{"cmd":"set_char_size","params":{"n":17}},{"cmd":"set_double_printing","params":[1]},{"cmd":"buffer","params":["0.02 Euro"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Biglietto Unico"]},{"cmd":"set_double_printing","params":[0]},{"cmd":"set_char_size","params":{"n":0}},{"cmd":"line_feed"},{"cmd":"buffer","params":["Non Rimborsabile"]},{"cmd":"line_feed"},{"cmd":"set_expanded_mode","params":{"n":0}},{"cmd":"set_barcode_width","params":{"n":2}},{"cmd":"print_barcode","params":{"m":5,"code":"54147992710810878899"}},{"cmd":"line_feed"},{"cmd":"buffer","params":["site name"]},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"total_cut"},{"cmd":"set_chars_upside_down","params":[0]}]'
    # text_lines_json = '[{"cmd":"init_printer"},{"cmd":"set_cpi","params":{"n":49}},{"cmd":"set_printing_mode","params":{"n":1}},{"cmd":"right_char_spacing","params":[0]},{"cmd":"set_chars_upside_down","params":[1]},{"cmd":"set_left_margin","params":{"nL":10,"nH":0}},{"cmd":"center_alignment"},{"cmd":"buffer","params":["Cod. Commerc"]},{"cmd":"buffer","params":["Cod. Commerc:"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["ARRIVEDERCI E GRAZIE"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Transazione autorizzata"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["totalToPay"]},{"cmd":"buffer","params":["IMPORTO EUR"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["cvmr"]},{"cmd":"buffer","params":["CVM"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["iad"]},{"cmd":"buffer","params":["IAD"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["ac"]},{"cmd":"buffer","params":["ARQC"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["tvr"]},{"cmd":"buffer","params":["TVR"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["un"]},{"cmd":"buffer","params":["UN"]},{"cmd":"buffer","params":["TrCC"]},{"cmd":"buffer","params":["TrCC"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["tt"]},{"cmd":"buffer","params":["TT"]},{"cmd":"buffer","params":["tcc"]},{"cmd":"buffer","params":["TCC"]},{"cmd":"buffer","params":["atc"]},{"cmd":"buffer","params":["ATC"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["appl_label"]},{"cmd":"buffer","params":["APPL"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["aid"]},{"cmd":"buffer","params":["A. ID"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["****"]},{"cmd":"buffer","params":["SCAD"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["pan"]},{"cmd":"buffer","params":["PAN"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Authorization Response Code"]},{"cmd":"buffer","params":["AUTH. RESP. CODE"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["boh"]},{"cmd":"buffer","params":["OPER."]},{"cmd":"buffer","params":["arc"]},{"cmd":"buffer","params":["AUT."]},{"cmd":"line_feed"},{"cmd":"buffer","params":["transaction type"]},{"cmd":"buffer","params":["CTLS"]},{"cmd":"buffer","params":["ops"]},{"cmd":"buffer","params":["Mod."]},{"cmd":"line_feed"},{"cmd":"buffer","params":["stan"]},{"cmd":"buffer","params":["STAN"]},{"cmd":"buffer","params":[" Terminal Identifier"]},{"cmd":"buffer","params":["TML"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["11:03"]},{"cmd":"buffer","params":["Ora"]},{"cmd":"buffer","params":["11/12/23"]},{"cmd":"buffer","params":["Data"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Acquirer Id"]},{"cmd":"buffer","params":["A. I. I. C."]},{"cmd":"line_feed"},{"cmd":"buffer","params":[" Merchant Identifier"]},{"cmd":"buffer","params":["Eserc."]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Receipt rows"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["ACQUISTO"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Acquire Name"]},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"total_cut"},{"cmd":"set_chars_upside_down","params":[0]}]'
    # load json from file
    with open("ricevuta_commands.json") as f:
        text_lines_json = f.read()
    with open("123456789.json") as f:
        receipt = json.load(f)
    # Parse the JSON string into a list of dictionaries
    header, row1, row2, courtesy, footer = parse_receipt_string(receipt["Receipt rows"])
    text_lines = json.loads(text_lines_json)
    text_lines[6]["params"][0] = header
    text_lines[15]["params"][0] = str(
        padded_cents_to_euros(receipt["Approved or Authorized Amount"])
    )
    text_lines[19]["params"][0] = receipt["CVMR"]
    text_lines[23]["params"][0] = receipt["IAD"]
    text_lines[27]["params"][0] = receipt["AC"]
    text_lines[31]["params"][0] = receipt["TVR"]
    text_lines[35]["params"][0] = receipt["UN"]
    text_lines[39]["params"][0] = receipt["TrCC"]
    text_lines[43]["params"][0] = receipt["TT"]
    text_lines[47]["params"][0] = receipt["TCC"]
    text_lines[51]["params"][0] = receipt["ATC"]
    text_lines[55]["params"][0] = receipt["APPL_LABEL"]
    text_lines[59]["params"][0] = receipt["AID"]
    text_lines[67]["params"][0] = receipt["PAN"]
    # Authorization Response Code
    text_lines[71]["params"][0] = receipt["Authorization Response Code"]
    text_lines[79]["params"][0] = receipt["ARC"]
    text_lines[83]["params"][0] = "ICC"
    text_lines[87]["params"][0] = receipt["OPS"]
    text_lines[91]["params"][0] = receipt["STAN"]
    text_lines[95]["params"][0] = receipt["Terminal Identifier"]
    parsed_datetime = datetime.strptime(
        receipt["Transaction Date and Time"], "%d%m%y%H%M%S"
    )
    text_lines[99]["params"][0] = parsed_datetime.strftime("%Y-%m-%d %H:%M:%S")
    text_lines[99]["params"][0] = receipt["Acquirer ID"]
    text_lines[103]["params"][0] = receipt["Merchant Identifier"]
    text_lines[107]["params"][0] = courtesy
    text_lines[109]["params"][0] = "Transazione autorizzata"

    # product = {
    #     "entranceDate": "2023-10-31",
    #     "entranceHour": "09:00:00",
    #     "siteName": "ANTIQUARIUM E PARCO ARCHEOLOGICO CANNE DELLA BATTAGLIA",
    #     "productName": "Intero",
    #     "qrCode": "abcdefghijz12345",
    #     "qty": 1,
    #     "price": 10,
    #     "validUntil": "2023-10-31 23:59:59",
    #     "additionalCode": "abcde",
    #     "holderName": "",
    #     "printedAt": "2023-10-31 16:01:01",
    # }
    # # replace parameter of index 16
    # text_lines[25]["params"][0] = f"{product['price']:.2f} Euro"
    # text_lines[35]["params"] = {"m": 8, "code": "{B" + product["qrCode"]}
    # text_lines[10]["params"][0] = product["validUntil"]
    # text_lines[15]["params"][0] = product["printedAt"]
    # text_lines[37]["params"][0] = product["siteName"]

    printer = Printer("/dev/ttyUSB0")
    print(f"Stampante pronta: {'SI' if printer.can_print() else 'NO'}")
    printer.print_lines_array(text_lines)
    printer.close()
