import paho.mqtt.client as mqtt
import time


if __name__ == "__main__":
    client = mqtt.Client()
    # set username and password
    client.username_pw_set("test", "test")
    client.connect("localhost", 1883, 60)

    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("cock")
        time.sleep(5)
        client.publish("cock", "Hello world!")
        print("Message published")

    client.on_connect = on_connect

    client.loop_start()
    time.sleep(10)
