import time
import paho.mqtt.client as mqtt
import minimalmodbus

# MQTT Broker settings
MQTT_BROKER = "5.196.23.212"  # Replace with your MQTT broker address
MQTT_PORT = 1883  # Replace with your MQTT broker port
MQTT_TOPIC_PUBLISH = "measures"  # Replace with your MQTT topic to publish to
devices = [{"port": "COM6", "registers": {""}}]


def usb_setup() -> list[minimalmodbus.Instrument]:
    """
    Sets up the USB connection to the sensors and returns a list of handles to them.
    """
    device_handles = []
    for device in devices:
        sensor = minimalmodbus.Instrument(device, 1)  # Change to your serial port
        sensor.serial.baudrate = 9600
        sensor.serial.bytesize = 8
        sensor.serial.parity = minimalmodbus.serial.PARITY_NONE
        sensor.serial.stopbits = 1
        sensor.serial.timeout = 1
        sensor.mode = minimalmodbus.MODE_RTU
        device_handles.append(sensor)
    return device_handles


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
    sensors = usb_setup()
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
        try:
            nitrogen_content = sensor.read_register(0x12, 0, 3, signed=False)
        except Exception as e:
            print(e)
            quit()
        try:
            client.publish(MQTT_TOPIC_PUBLISH, str(nitrogen_content))
        except Exception as e:
            print(e)
