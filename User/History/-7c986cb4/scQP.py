import paho.mqtt.client as mqtt


class mqtt_client:
    def __init__(self, company_id) -> None:
        client = mqtt.Client()
        client.connect("localhost", 1883, 60)
        # on_message
        client.subscribe("kit/")

        self.messages = []

        def on_message(client, userdata, message):
            msg = str(message.payload.decode("utf-8"))
            print("message received ", msg)
            print("message topic=", message.topic)
            print("message qos=", message.qos)
            print("message retain flag=", message.retain)

            self.messages.append(msg)

        client.on_message = on_message
        client.loop_start()
