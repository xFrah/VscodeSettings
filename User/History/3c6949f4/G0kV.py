import asyncio
import json
import math
import threading
import time

import asyncio_mqtt
import cv2
import numpy as np


class my_mqtt_client:
    def __init__(self, host="localhost", topic="humidity/command", client_id="MQTT_CLIENT_" + str(time.time())):
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
        self.rx_func = lambda x: 
        self.rx_buffer_size = 1
        self.thread = threading.Thread(target=self.mqtt_thread)
        self.thread.start()

    def decode(self, payload):
        try:
            return json.loads(payload)
        except Exception:
            return None

    def start(self):
        self.stopped = False

    def stop(self):
        self.stopped = True

    def data_ready(self):
        return len(self.frame_buffer) > 0

    def get_frame(self):
        with self.frame_buffer_lock:
            return self.frame_buffer.pop(0)

    def send_frame(self, topic, frame):
        with self.to_send_buffer_lock:
            self.to_send_buffer.append((topic, frame))

    def buffer_frame(self, frame):
        with self.frame_buffer_lock:
            self.frame_buffer.append(frame)
            del self.frame_buffer[: -self.rx_buffer_size]

    def mqtt_thread(self):
        while True:

            async def receiver(client):
                async with client.messages() as messages:
                    await client.subscribe(self.topic)
                    print("MQTT >> Subscribed.")
                    async for message in messages:
                        # print(message.payload)
                        if self.stopped:
                            print("MQTT >> Stopping the receiver.")
                            break
                        self.buffer_frame(message.payload)

                    print("MQTT >> Unsubscribing")

            async def sender(client):
                while True:
                    if self.stopped:
                        print("MQTT >> Stopping the sender.")
                        break
                    if len(self.to_send_buffer) > 0:
                        with self.to_send_buffer_lock:
                            topic, frame = self.to_send_buffer.pop(0)
                        print("MQTT >> Sending a frame: ", frame)
                        await client.publish(topic, frame)
                        await asyncio.sleep(0)
                        continue

                    # we only sleep if there is nothing to send to avoid wasting CPU
                    await asyncio.sleep(0.2)
                    # await client.publish(self.topic, "Hello world!")

            async def starter():
                async with asyncio_mqtt.Client(self.host, client_id=self.client_id) as client:
                    await asyncio.gather(receiver(client), sender(client))
                print("MQTT >> Client stopped.")

            if not self.stopped:
                try:
                    asyncio.run(starter())
                    print("MQTT >> Thread ended.")
                except asyncio_mqtt.error.MqttCodeError as e:
                    print("MQTT >> Client killed because host is no longer available.")
                except asyncio_mqtt.error.MqttError as e:
                    print(e)
                except Exception as e:
                    print(e)
            else:
                time.sleep(0.7)


def lidar_draw(packet, scaling_factor=0.05):
    angle_lower_limit_1, angle_upper_limit_1, angle_lower_limit_2, angle_upper_limit_2 = packet["limits"]

    img = np.zeros((1000, 1000, 3), np.uint8)

    # draw a circle in the center of the image
    cv2.circle(img, (500, 500), 5, (255, 255, 255), -1)

    scale = scaling_factor

    # draw a line for each distance
    for angle, distance in packet["points"]:
        # convert distance and angle to x and y coordinates
        x = 500 + (distance * scale) * math.cos(math.radians(angle))
        y = 500 + (distance * scale) * math.sin(math.radians(angle))
        # draw a line from the center to the distance
        cv2.circle(
            img,
            (int(x), int(y)),
            1,
            (0, 0, 255)
            if angle_lower_limit_1 <= angle <= angle_upper_limit_1 or angle_lower_limit_2 <= angle <= angle_upper_limit_2
            else (255, 0, 0),
            -1,
        )

    # detection_frame = []

    for person_angle, distance in packet["people"]:
        x = 500 + (distance * scale) * math.cos(math.radians(person_angle))
        y = 500 + (distance * scale) * math.sin(math.radians(person_angle))

        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 5)

    return img
