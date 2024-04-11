import paho.mqtt.client as mqtt


if __name__ == "__main__":
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)

    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("test")

    client.on_connect = on_connect

    client.loop_start()
