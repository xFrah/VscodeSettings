import time
import paho.mqtt.client as mqtt
import minimalmodbus
import threading

# Global connection object
global_instrument = None

# MQTT Broker settings
MQTT_BROKER = "your.broker.address"  # Replace with your MQTT broker address
MQTT_PORT = 1883  # Replace with your MQTT broker port
MQTT_TOPIC_SUBSCRIBE = (
    "your/subscribe/topic"  # Replace with your MQTT topic to subscribe to
)
MQTT_TOPIC_PUBLISH = "your/publish/topic"  # Replace with your MQTT topic to publish to


# Setup the Modbus connection
def setup_instrument():
    global global_instrument
    global_instrument = minimalmodbus.Instrument(
        "COM6", 1
    )  # Change to your serial port
    global_instrument.serial.baudrate = 9600
    global_instrument.serial.bytesize = 8
    global_instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
    global_instrument.serial.stopbits = 1
    global_instrument.serial.timeout = 1
    global_instrument.mode = minimalmodbus.MODE_RTU


# Function to handle incoming MQTT messages
def on_message(client, userdata, message):
    if message.topic == MQTT_TOPIC_SUBSCRIBE:
        payload = str(message.payload.decode("utf-8"))
        if payload == "measure":
            try:
                # Read the nitrogen content register
                nitrogen_content = global_instrument.read_register(
                    0x12, 0, 3, signed=False
                )
                # Publish the nitrogen content
                client.publish(
                    MQTT_TOPIC_PUBLISH, f"Nitrogen content: {nitrogen_content} mg/kg"
                )
            except Exception as e:
                print(e)
                client.publish(MQTT_TOPIC_PUBLISH, "Failed to read from instrument")


# Initialize MQTT client and set callbacks
def setup_mqtt_client():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC_SUBSCRIBE)
    return client


# Main function to start the MQTT loop in a separate thread
def start_mqtt_thread():
    client = setup_mqtt_client()

    def mqtt_loop():
        client.loop_forever()

    thread = threading.Thread(target=mqtt_loop)
    thread.start()


if __name__ == "__main__":
    setup_instrument()
    client = setup_mqtt_client()
    client.loop_start()
    while True:
        time.sleep(1)
