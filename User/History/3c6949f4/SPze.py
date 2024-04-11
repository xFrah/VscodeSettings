import threading
import asyncio_mqtt.

class my_mqtt_client:
    def __init__(self, host="localhost", topic="humidity/", client_id="MQTT_CLIENT_" + str(time.time())):
        self.thread: threading.Thread
        self.client: asyncio_mqtt.Client
        self.stopped = False
        self.frame_buffer = []
        self.to_send_buffer = []
        self.to_send_buffer_lock = threading.Lock()
        self.frame_buffer_lock = threading.Lock()
        self.host = host
        self.topic = topic
        self.client_id = client_id

    def decode(self, payload):
        try:
            return json.loads(payload)
        except Exception:
            return None

    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self.mqtt_thread)
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join()

    def data_ready(self):
        return len(self.frame_buffer) > 0

    def get_frame(self):
        with self.frame_buffer_lock:
            return self.frame_buffer.pop(0)

    def send_frame(self, frame):
        with self.to_send_buffer_lock:
            self.to_send_buffer.append(frame)

    def buffer_frame(self, frame):
        with self.frame_buffer_lock:
            self.frame_buffer.append(frame)

    def mqtt_thread(self):
        async def receiver(client):
            async with client.messages() as messages:
                await client.subscribe(self.topic)
                print("MQTT >> Subscribed.")
                async for message in messages:
                    print(message.payload)
                    if self.stopped:
                        print("MQTT >> Stopping the receiver.")
                        break
                    self.buffer_frame(message.payload)

                    message_dict = self.decode(self.get_frame())
                    if message_dict is not None:
                        print("MQTT >> Got a json frame: ", message_dict)
                    else:
                        print("MQTT >> Got a non-json message.")

                print("MQTT >> Unsubscribing")

        async def sender(client):
            while True:
                if self.stopped:
                    print("MQTT >> Stopping the sender.")
                    break
                if len(self.to_send_buffer) > 0:
                    with self.to_send_buffer_lock:
                        frame = self.to_send_buffer.pop(0)
                    print("MQTT >> Sending a frame: ", frame)
                    await client.publish(self.topic, "Hello world!")
                    continue

                # we only sleep if there is nothing to send to avoid wasting CPU
                await asyncio.sleep(0.2)
                await client.publish(self.topic, "Hello world!")

        async def starter():
            async with asyncio_mqtt.Client(self.host, client_id=self.client_id) as client:
                await asyncio.gather(receiver(client), sender(client))
            print("MQTT >> Client stopped.")

        asyncio.run(starter())
        print("MQTT >> Thread ended.")