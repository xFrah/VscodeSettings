import paho.mqtt.client as mqtt
import threading


class mqtt_client:
    def __init__(self) -> None:
        client = mqtt.Client()

        def on_message(client, userdata, message):
            msg = str(message.payload.decode("utf-8"))
            print("message received ", msg)
            # print("message topic=", message.topic)
            # print("message qos=", message.qos)
            # print("message retain flag=", message.retain)
            with self.lock:
                self.messages.append(msg)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("connected OK Returned code=", rc)
            else:
                print("Bad connection Returned code=", rc)

        client.on_message = on_message
        client.on_connect = on_connect
        client.connect("localhost", 1883, 60)
        client.subscribe(f"kits")
        self.lock = threading.Lock()
        self.messages = []
        client.loop_start()

    def data_ready(self):
        return len(self.messages) > 0

    def get_message(self):
        with self.lock:
            return self.messages.pop(0) if self.messages else None
