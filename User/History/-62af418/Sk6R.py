import json
import time
import paho.mqtt.client as mqtt
import minimalmodbus

# MQTT Broker settings
MQTT_BROKER = "5.196.23.212"  # Replace with your MQTT broker address
MQTT_PORT = 1883  # Replace with your MQTT broker port
MQTT_TOPIC_PUBLISH = "measures"  # Replace with your MQTT topic to publish to
MQTT_TOPIC_SUBSCRIBE = "commands"
device = "COM6"
registers = {"humidity": 0x12}


def usb_setup() -> minimalmodbus.Instrument:
    """
    Sets up the USB connection to the sensors and returns a list of handles to them.
    """
    sensor = minimalmodbus.Instrument(device, 1)  # Change to your serial port
    sensor.serial.baudrate = 9600
    sensor.serial.bytesize = 8
    sensor.serial.parity = minimalmodbus.serial.PARITY_NONE
    sensor.serial.stopbits = 1
    sensor.serial.timeout = 1
    sensor.mode = minimalmodbus.MODE_RTU
    return sensor


def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    if rc == 0:
        client.subscribe(MQTT_TOPIC_PUBLISH)


def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_disconnect(client, userdata, rc):
    print("Disconnected with result code " + str(rc))


if __name__ == "__main__":
    # sensor = usb_setup()
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe
    client.on_disconnect = on_disconnect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC_PUBLISH)
    client.loop_start()
    while True:
        time.sleep(0.5)
        res = {}
        for reg_name, reg_address in registers.items():
            try:
                res[reg_name] = sensor.read_register(reg_address, 0, 3, signed=False)
            except Exception as e:
                res[reg_name] = None
        try:
            client.publish(MQTT_TOPIC_PUBLISH, json.dumps(res).encode("utf-8"))
        except Exception as e:
            print(e)
