import paho.mqtt.client as mqtt
import threading


class mqtt_client:
    def __init__(self, company_id) -> None:
        client = mqtt.Client()
        client.connect("localhost", 1883, 60)
        client.subscribe(f"kit/{company_id}/#")
        self.lock = threading.Lock()
        self.messages = list()

        def on_message(client, userdata, message):
            msg = str(message.payload.decode("utf-8"))
            print("message received ", msg)
            print("message topic=", message.topic)
            print("message qos=", message.qos)
            print("message retain flag=", message.retain)
            with self.lock:
                self.messages.append(msg)

        client.on_message = on_message
        client.loop_start()

    def data_ready(self):
        return len(self.messages) > 0

    def get_message(self):
        with self.lock:
            return self.messages.pop(0) if self.messages else None
