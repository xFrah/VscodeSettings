# import mqtt
import json
import time

import paho.mqtt.client as mqtt

username, password = "aditus", "Aditus2023!"
topic = "device/bitonto"
host, port = "34.154.15.151", 1883

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(username, password)
mqtt_client.connect(host, 1883, 60)


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    mqtt_client.subscribe(topic)


def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))


mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.loop_start()

packets = [
    {
        "packet_type": "server-pos",
        "seq_num": "1",
        "transaction_id": "123456789",
        "language": "it",
        "date": "2023-10-31",
        "hour": "09:00:00",
        "products": [
            {"id": 250, "price": 10, "qty": 2},
            {"id": 240, "price": 5, "qty": 1},
        ],
        "tvm_machine": {
            "id": 2,
            "site_id": 18,
            "code": "022531727",
            "localMachineIp": "0.0.0.0",
            "term_id": "92541182",
            "printer_id": "KUBE200",
        },
        "cart_id": 1080,
        "totalToPay": 0.1,
    },
    {
        "packet_type": "server-stampante",
        "seq_num": "1",
        "transaction_id": "1",
        "language": "it",
        "products": [
            {
                "entranceDate": "2023-10-31",
                "entranceHour": "09:00:00",
                "siteName": "ANTIQUARIUM E PARCO ARCHEOLOGICO CANNE DELLA BATTAGLIA",
                "productName": "Intero",
                "qrCode": "abcdefgz12345",
                "qty": 1,
                "price": 0.1,
                "validUntil": "2023-10-31 23:59:59",
                "additionalCode": "abcde",
                "holderName": "",
                "printedAt": "2023-10-31 16:01:01",
            },
            {
                "entranceDate": "2023-10-31",
                "entranceHour": "09:00:00",
                "siteName": "ANTIQUARIUM E PARCO ARCHEOLOGICO CANNE DELLA BATTAGLIA",
                "productName": "Intero",
                "qrCode": "abcdefgvyz67",
                "qty": 1,
                "price": 0.1,
                "validUntil": "2023-10-31 23:59:59",
                "additionalCode": "fgkijk",
                "holderName": "",
                "printedAt": "2023-10-31 16:01:01",
            },
        ],
    },
]

for packet in packets[1:]:
    i = input("Press enter to send packet")
    mqtt_client.publish(topic, json.dumps(packet))
    time.sleep(5)
