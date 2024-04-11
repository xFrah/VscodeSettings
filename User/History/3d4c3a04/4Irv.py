import paho.mqtt.client as mqtt
import time


if __name__ == "__main__":
    client = mqtt.Client("asdasdasd")
    # set username and password
    # change client id
    client.connect("test.mosquitto.org", 1883, 60)

    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("cock")
        time.sleep(5)
        client.publish("cock", "Hello world!")
        print("Message published")

    def on_message(client, userdata, msg):
        print("Received message: " + msg.topic + " " + str(msg.payload))

    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_forever()
    time.sleep(10)
