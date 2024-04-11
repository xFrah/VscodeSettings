from datetime import datetime
import paho.mqtt.client as mqtt
import json
import time
import queue

import serial
import packets
from printer import Printer, padded_cents_to_euros, parse_receipt_string
from pos import StartOperation, activate_pos, receive_standard_pos, SendAmount
from helpers import check_license, debug, get_configuration, save_receipt, get_receipt


max_pkt_age = 10
pos_timeout = 60
init_pos_timeout = 10
last_transaction = time.time()
rebounce_time = 15


def mqtt_send(msg: dict) -> None:
    """
    Sends a MQTT message to the server.
    """
    msg_ = json.dumps(msg)
    mqtt_client.publish(topic, msg_)
    debug("Sent: " + msg_)


def receive() -> None:
    """
    In this state we are waiting for the server to initiate a transaction.
    """
    msg, timestamp = mqtt_buffer.get(block=True, timeout=10)
    if time.time() - timestamp > max_pkt_age:
        return debug(f"Packet is too old: {time.time() - timestamp}s > {max_pkt_age}s")
    msg = json.loads(msg)
    debug("Received: " + str(msg))
    packet_type = msg["packet_type"]
    if packet_type == "server-pos":
        mqtt_send(packets.ack(msg, "TRANSACTION"))
        debug("Received server-pos packet, changing status to TRANSACTION")
        transaction(msg)
    elif packet_type == "server-stampante":
        mqtt_send(packets.ack(msg, "PRINTING"))
        products = msg["products"]
        for product in products:
            printing(product, msg["transaction_id"])
            time.sleep(0.5)
        debug("Received server-stampante packet, changing status to PRINTING")
    elif packet_type == "server-ricevuta":
        mqtt_send(packets.ack(msg, "RECEIPT"))
        debug("Received server-ricevuta packet, sending receipt")
        send_receipt(msg["transaction_id"])


def transaction(msg: dict):
    """
    In this state we write to the pos and execute the transaction.
    """
    # check if pos is sending any messages
    # wait until we are 15 seconds without messages
    global last_transaction
    while time.time() - last_transaction < rebounce_time:
        debug(
            f"Waiting for rebounce, {time.time() - last_transaction:.02f}s < {rebounce_time:.02f}s"
        )
        time.sleep(1)
    debug("Starting transaction")

    start = time.time()
    buf = b""

    pay = StartOperation(
        config["terminal_id"], cmd=StartOperation.Payment, amount=msg["totalToPay"]
    )
    retries = config["payment_init_retries"]
    trials = 0
    success = False
    try:
        POS = serial.Serial(config["pos_port"], 115200, timeout=1)
        debug("POS opened")
        while trials < retries and not success:
            debug(f"Sending payment, trial {trials + 1} of {retries}.")
            POS.write(pay.get_string_to_send().encode())
            start = time.time()
            while time.time() - start < 8:
                time.sleep(1)
                tbuf = POS.read_all()
                if not tbuf:
                    continue
                debug(str(tbuf))
                buf += tbuf
                if b"INIZIO TRANSAZIONE" in buf:
                    success = True
                    debug("Transaction started")
                    last_transaction = time.time()
                    break
            trials += 1
    except Exception as e:
        try:
            POS.close()
        except:
            debug("POS not open")
        print(e)

    mqtt_send(
        packets.pos_server1(
            transaction_id=msg["transaction_id"],
            success=success,
            cart_id=msg["cart_id"],
        )
    )

    if not success:
        return debug("Transaction failed")

    start = time.time()
    success = False
    try:
        while time.time() - start < pos_timeout:
            time.sleep(1)
            packet = receive_standard_pos(POS, timeout=10)
            if packet is None:
                continue
            if packet["Command Code"] == "E":
                # get result
                res = packet["Transaction Result"]
                success = True if res == "Approved" else False
                print("RISULTATO", res)
                break
            elif packet["Command Code"] == "F":
                success = False
                break
            elif packet["Command Code"] == "I":
                sa = SendAmount(msg["totalToPay"], config["terminal_id"])
                c = sa.get_string_to_send().encode()
                POS.write(c)
                print(c)
    except Exception as e:
        success = False
        print(e)
    finally:
        try:
            POS.close()
        except:
            debug("POS not open")

    mqtt_send(
        packets.pos_server2(
            transaction_id=msg["transaction_id"],
            success=success,
        )
    )

    if success and packet is not None:
        save_receipt(packet, msg["transaction_id"])


def printing(product: dict, transaction_id: str) -> None:
    """
    We try printing.
    """
    text_lines_json = '[{"cmd":"init_printer"},{"cmd":"set_cpi","params":{"n":49}},{"cmd":"set_printing_mode","params":{"n":1}},{"cmd":"right_char_spacing","params":[0]},{"cmd":"set_chars_upside_down","params":[1]},{"cmd":"set_left_margin","params":{"nL":10,"nH":0}},{"cmd":"center_alignment"},{"cmd":"line_spacing","params":[24]},{"cmd":"buffer","params":["Valido fino a "]},{"cmd":"set_double_printing","params":[1]},{"cmd":"buffer","params":["giorno scadenza"]},{"cmd":"set_double_printing","params":[0]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Emesso il "]},{"cmd":"set_double_printing","params":[1]},{"cmd":"buffer","params":["08/06/17"]},{"cmd":"set_double_printing","params":[0]},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"set_left_margin","params":{"nL":5,"nH":0}},{"cmd":"buffer","params":["Esente IVA Art.10 C.1 n.22 DPR 633/72 SM"]},{"cmd":"set_left_margin","params":{"nL":10,"nH":0}},{"cmd":"line_feed"},{"cmd":"set_char_size","params":{"n":17}},{"cmd":"set_double_printing","params":[1]},{"cmd":"buffer","params":["0.02 Euro"]},{"cmd":"line_feed"},{"cmd":"buffer","params":["Biglietto Unico"]},{"cmd":"set_double_printing","params":[0]},{"cmd":"set_char_size","params":{"n":0}},{"cmd":"line_feed"},{"cmd":"buffer","params":["Non Rimborsabile"]},{"cmd":"line_feed"},{"cmd":"set_expanded_mode","params":{"n":0}},{"cmd":"set_barcode_width","params":{"n":2}},{"cmd":"print_barcode","params":{"m":5,"code":"54147992710810878899"}},{"cmd":"line_feed"},{"cmd":"buffer","params":["site name"]},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"line_feed"},{"cmd":"total_cut"},{"cmd":"set_chars_upside_down","params":[0]}]'
    text_lines = json.loads(text_lines_json)

    text_lines[25]["params"][0] = f"{product['price']:.2f} Euro"
    text_lines[35]["params"] = {"m": 8, "code": "{B" + product["qrCode"]}
    text_lines[10]["params"][0] = product["validUntil"]
    text_lines[15]["params"][0] = product["printedAt"]
    text_lines[37]["params"][0] = product["siteName"]
    text_lines[27]["params"][0] = f"Biglietto {product['productName']}"

    printer.print_lines_array(text_lines)
    time.sleep(1)
    success = printer.can_print()
    printer_status = printer.dump_printer_status_dict()
    mqtt_send(packets.stampante_server(transaction_id, success, printer_status))
    debug(f"Printing: {'Success' if success else 'Failed'}")


def printing_receipt(receipt, transaction_id: str) -> None:
    """
    Sends the receipt to the server.
    """
    receipt = get_receipt(transaction_id)
    success = False
    if receipt is not None:
        text_lines_json = '[    {        "cmd": "init_printer"    },    {        "cmd": "set_cpi",        "params": {            "n": 49        }    },    {        "cmd": "set_printing_mode",        "params": {            "n": 1        }    },    {        "cmd": "right_char_spacing",        "params": [            0        ]    },    {        "cmd": "set_left_margin",        "params": {            "nL": 10,            "nH": 0        }    },    {        "cmd": "center_alignment"    },    {        "cmd": "buffer",        "params": [            "Header"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "buffer",        "params": [            "Row1"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "buffer",        "params": [            "Row2"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "IMPORTO EUR:  "        ]    },    {        "cmd": "buffer",        "params": [            "totalToPay"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "CVM:  "        ]    },    {        "cmd": "buffer",        "params": [            "cvmr"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "IAD:  "        ]    },    {        "cmd": "buffer",        "params": [            "iad"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "ARQC:  "        ]    },    {        "cmd": "buffer",        "params": [            "ac"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "TVR:  "        ]    },    {        "cmd": "buffer",        "params": [            "tvr"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "UN:  "        ]    },    {        "cmd": "buffer",        "params": [            "un"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "TrCC:  "        ]    },    {        "cmd": "buffer",        "params": [            "TrCC"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "TT:  "        ]    },    {        "cmd": "buffer",        "params": [            "tt"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "TCC:  "        ]    },    {        "cmd": "buffer",        "params": [            "tcc"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "ATC:  "        ]    },    {        "cmd": "buffer",        "params": [            "atc"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "APPL:  "        ]    },    {        "cmd": "buffer",        "params": [            "appl_label"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "A. ID:  "        ]    },    {        "cmd": "buffer",        "params": [            "aid"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "SCAD:  "        ]    },    {        "cmd": "buffer",        "params": [            "****"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "PAN:  "        ]    },    {        "cmd": "buffer",        "params": [            "pan"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "AUTH. RESP. CODE:  "        ]    },    {        "cmd": "buffer",        "params": [            "Authorization Response Code"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "OPER.:  "        ]    },    {        "cmd": "buffer",        "params": [            "*"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "AUT.:  "        ]    },    {        "cmd": "buffer",        "params": [            "arc"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "CTLS:  "        ]    },    {        "cmd": "buffer",        "params": [            "CCI"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "Mod.:  "        ]    },    {        "cmd": "buffer",        "params": [            "ops"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "STAN:  "        ]    },    {        "cmd": "buffer",        "params": [            "stan"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "TML:  "        ]    },    {        "cmd": "buffer",        "params": [            "Terminal Identifier"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "Data:  "        ]    },    {        "cmd": "buffer",        "params": [            "11/12/23"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "A. I. I. C.:  "        ]    },    {        "cmd": "buffer",        "params": [            "Acquirer Id"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "left_alignment"    },    {        "cmd": "buffer",        "params": [            "Eserc.:  "        ]    },    {        "cmd": "buffer",        "params": [            "Merchant Identifier"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "center_alignment"    },    {        "cmd": "line_feed"    },    {        "cmd": "buffer",        "params": [            "ARRIVEDERCI E GRAZIE"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "buffer",        "params": [            "TRANSAZIONE AUTORIZZATA"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "buffer",        "params": [            "Foote notes"        ]    },    {        "cmd": "line_feed"    },    {        "cmd": "line_feed"    },    {        "cmd": "line_feed"    },    {        "cmd": "line_feed"    },    {        "cmd": "line_feed"    },    {        "cmd": "total_cut"    }]'
        header, row1, row2, courtesy, footer = parse_receipt_string(receipt["Receipt rows"])
        text_lines = json.loads(text_lines_json)
        text_lines[6]["params"][0] = header
        text_lines[8]["params"][0] = row1
        text_lines[10]["params"][0] = row2
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
        text_lines[103]["params"][0] = receipt["Acquirer ID"]
        text_lines[107]["params"][0] = receipt["Merchant Identifier"]
        text_lines[111]["params"][0] = courtesy
        text_lines[113]["params"][0] = "Transazione autorizzata"
        text_lines[115]["params"][0] = footer
        printer.print_lines_array(text_lines)
        success = printer.can_print()
        success = True
    mqtt_send(packets.ricevuta_server(transaction_id, receipt, success))
    if receipt is None:
        debug("No receipt found")
    else:
        debug("Receipt sent to server" if success else "Receipt not printed")


def main():
    def on_connect(client, userdata, flags, rc):
        conn_errors = {
            0: "Connection successful",
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorised",
        }
        debug("Connected with result: " + conn_errors.get(rc, "Unknown error"))
        check_license()
        mqtt_client.subscribe(topic)
        # mqtt_client.publish("test", "Hello World!")

    def on_disconnect(client, userdata, rc):
        debug("Disconnected with result code " + str(rc))

    def on_subscribe(client, userdata, mid, granted_qos):
        debug(f"Subscribed to topic {topic}")

    def on_message(client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))
        try:
            mqtt_buffer.put((msg.payload.decode(), time.time()), block=False)
        except Exception as e:
            debug("Error putting message in buffer " + str(e))

    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_subscribe = on_subscribe
    mqtt_client.on_message = on_message

    mqtt_client.username_pw_set(username, password)
    mqtt_client.connect(host, port, 60)
    mqtt_client.loop_start()

    while True:
        try:
            receive()
        except queue.Empty:
            pass
        except json.decoder.JSONDecodeError as e:
            debug(f"Error decoding json: {e}")
        except KeyError as e:
            debug(f"Json packet is missing a field: {e}")
        debug("Main loop")
        time.sleep(0.1)


if __name__ == "__main__":
    while True:
        try:
            config = get_configuration()
            mqtt_client = mqtt.Client()
            mqtt_buffer = queue.Queue()
            printer = Printer(config["printer_port"])
            activate_pos()
            username, password = config["mqtt_username"], config["mqtt_password"]
            topic = config["mqtt_topic"]
            host, port = config["mqtt_host"], config["mqtt_port"]
            main()
        except Exception as e:
            print(e)
        try:
            mqtt_client.disconnect()
        except Exception as e:
            print(e)
        try:
            mqtt_client.loop_stop()
        except Exception as e:
            print(e)
        time.sleep(5)
