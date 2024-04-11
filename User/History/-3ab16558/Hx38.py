import time
import paho.mqtt.client as mqtt
import minimalmodbus

# MQTT Broker settings
MQTT_BROKER = ""  # Replace with your MQTT broker address
MQTT_PORT = 1883  # Replace with your MQTT broker port
MQTT_TOPIC_PUBLISH = "measures"  # Replace with your MQTT topic to publish to


if __name__ == "__main__":
    sensor = minimalmodbus.Instrument("COM6", 1)  # Change to your serial port
    sensor.serial.baudrate = 9600
    sensor.serial.bytesize = 8
    sensor.serial.parity = minimalmodbus.serial.PARITY_NONE
    sensor.serial.stopbits = 1
    sensor.serial.timeout = 1
    sensor.mode = minimalmodbus.MODE_RTU
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    while True:
        time.sleep(1)
        try:
            # Read the nitrogen content register
            nitrogen_content = sensor.read_register(0x12, 0, 3, signed=False)
            # Publish the nitrogen content
            client.publish(
                MQTT_TOPIC_PUBLISH, f"Nitrogen content: {nitrogen_content} mg/kg"
            )
        except Exception as e:
            print(e)
            client.publish(MQTT_TOPIC_PUBLISH, "Failed to read from instrument")
