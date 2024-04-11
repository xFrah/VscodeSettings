import paho.mqtt.client as mqtt


class mqtt_client:
    def __init__(self) -> None:
        client = mqtt.Client()
        client.connect("localhost", 1883, 60)
        # on_message
        client.subscribe("kit/#")
        client.on_message = on_message
        client.loop_start()

def on_message(client, userdata, message):
msg = str(message.payload.decode("utf-8"))
print("message received ", msg)
print("message topic=", message.topic)
print("message qos=", message.qos)
print("message retain flag=", message.retain)

# protocol
# opmode = chars 0 to 1
# mac = chars 2 to 18
# data = chars 19 to end

opmode = msg[0:2]
mac = msg[2:18]
data = msg[19:]

if opmode not in opmodes:
    print("Unknown opmode.")
if not db.check_if_valid_mac(mac):
    print("Invalid MAC address.")

if opmode == _STATUS_REPORT:
    print("Status report received.")
    db.save_report(mac, data)
