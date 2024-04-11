import customtkinter
import threading
from PIL import Image
import paho.mqtt.client as mqtt
import cv2
from amqtt.mqtt.constants import QOS_0, QOS_1, QOS_2
import asyncio
from amqtt.broker import Broker
from amqtt.client import MQTTClient
import logging
import asyncio_mqtt
import os


def mqtt_thread(app):
    logger = logging.getLogger(__name__)
    asyncio.set_event_loop(asyncio.new_event_loop())

    config = {
        "listeners": {
            "default": {
                "type": "tcp",
                "bind": "127.0.0.1:1883",
            },
            "ws-mqtt": {
                "bind": "127.0.0.1:8080",
                "type": "ws",
                "max_connections": 10,
            },
        },
        "sys_interval": 10,
        "auth": {
            "allow-anonymous": False,
            "password-file": os.path.join(os.path.dirname(os.path.realpath(__file__)), "passwd.txt"),
            "plugins": ["auth_file", "auth_anonymous"],
        },
        "topic-check": {"enabled": False},
    }

    broker = Broker(config)

    async def test_coro():
        await broker.start()
        # await asyncio.sleep(5)
        # await broker.shutdown()

    async def test_client_subscribe_publish(broker):
        sub_client = MQTTClient("client-1")
        await sub_client.connect("mqtt://localhost/")

        await sub_client.publish("/qos0", b"data", QOS_0)

        await sub_client.publish("/qos2", b"data", QOS_2)

        await sub_client.subscribe(
            [
                ("cock", QOS_1),
            ]
        )

        await asyncio.sleep(0.1)
        while True:
            await sub_client.publish("cock", b"data", QOS_1)
            message = await sub_client.deliver_message()
            if message is not None:
                print(message.publish_packet.payload)

    async def test_coro2():
        await asyncio.sleep(10)
        async with asyncio_mqtt.Client("localhost") as client:
            print("subscribing")
            async with client.messages() as messages:
                await client.subscribe("humidity/")
                asyncio.create_task(timer(client))
                async for message in messages:
                    print(message.payload)
                print("unsubscribing")

    async def timer(client):
        while True:
            await asyncio.sleep(10)
            await client.publish("humidity/", "Hello world!")

    formatter = "[%(asctime)s] :: %(levelname)s :: %(message)s"
    # formatter = "%(asctime)s :: %(levelname)s :: %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)
    asyncio.get_event_loop().run_until_complete(test_coro())
    #asyncio.get_event_loop().run_until_complete(test_client_subscribe_publish(broker))
    # asyncio.get_event_loop().run_until_complete(test_coro2())
    asyncio.get_event_loop().run_forever()


def lidar_draw(lidar, distances, slices, people_angles, angle_lower_limit_1, angle_upper_limit_1, angle_lower_limit_2, angle_upper_limit_2):
    if distances and slices:
        # create a black image
        img = np.zeros((1000, 1000, 3), np.uint8)

        # draw a circle in the center of the image
        cv2.circle(img, (500, 500), 5, (255, 255, 255), -1)

        scale = 0.05

        # draw a line for each distance
        for angle, distance in distances:
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

        for person_angle, _ in people_angles:
            distance, _ = lidar.get_distance_at_angle(slices, person_angle)
            # convert distance and angle to x and y coordinates
            x = 500 + (distance * scale) * math.cos(math.radians(person_angle))
            y = 500 + (distance * scale) * math.sin(math.radians(person_angle))
            # draw a line from the center to the distance
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 5)
            # detection_frame.append((slice_index, rect, distance))

        # for slice_ in slices:
        #     medium_angle = sum([angle for angle, distance in slice_]) / len(slice_)
        #     medium_distance = min([distance for angle, distance in slice_ if distance > 30])
        #     # draw a line from the center to the distance
        #     cv2.line(img, (500, 500),
        #              (int(500 + (medium_distance * scale) * math.cos(math.radians(medium_angle))), int(500 + (medium_distance * scale) * math.sin(math.radians(medium_angle)))),
        #              (255, 255, 255), 1)

        # draw a purple circle at the angle

        # print(f"Detect >> Elapsed time for lidar: {(time.time() - start) * 1E3:.01f}ms")
        cv2.imshow("Lidar", img)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("image_example.py")
        self.geometry("700x700")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        self.logo_image = customtkinter.CTkImage(Image.open("paper.png"), size=(26, 26))
        self.image = customtkinter.CTkImage(Image.open("paper.png"), size=(350, 350))
        self.home_image = customtkinter.CTkImage(
            light_image=Image.open("paper.png"),
            dark_image=Image.open("paper.png"),
            size=(20, 20),
        )
        self.chat_image = customtkinter.CTkImage(
            light_image=Image.open("paper.png"),
            dark_image=Image.open("paper.png"),
            size=(20, 20),
        )
        self.add_user_image = customtkinter.CTkImage(
            light_image=Image.open("paper.png"),
            dark_image=Image.open("paper.png"),
            size=(20, 20),
        )

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(
            self.navigation_frame,
            text="  Image Example",
            image=self.logo_image,
            compound="left",
            font=customtkinter.CTkFont(size=15, weight="bold"),
        )
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.frame_1_button = customtkinter.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Lidar + AI",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=self.home_button_event,
        )
        self.frame_1_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Interpolated Lidar",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=self.frame_2_button_event,
        )
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Camera",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=self.frame_3_button_event,
        )
        self.frame_3_button.grid(row=3, column=0, sticky="ew")

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_frame_large_image_label = customtkinter.CTkLabel(self.home_frame, text="", image=self.image)
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=40)

        def callback_scaling(value):
            self.slider_1_label.configure(text=f"Scaling: {int(value)/100}")

        # label the slider with a text
        self.slider_1_label = customtkinter.CTkLabel(
            self.home_frame,
            text="Scaling: 2.50",
        )
        self.slider_1_label.grid(sticky="w", padx=45)

        self.slider_1 = customtkinter.CTkSlider(master=self.home_frame, from_=0, to=500, width=400, command=callback_scaling)
        self.slider_1.grid(pady=10)
        self.slider_1.set(250)

        # create second frame
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # create third frame
        self.third_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # select default frame
        self.select_frame_by_name("Lidar + AI")

    def update_image(self, image: Image.Image):
        self.home_frame_large_image_label.configure(image=customtkinter.CTkImage(image, size=(350, 350)))

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.frame_1_button.configure(fg_color=("gray75", "gray25") if name == "Lidar + AI" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "Interpolated Lidar" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "Camera" else "transparent")

        # show selected frame
        if name == "Lidar + AI":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "Interpolated Lidar":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "Camera":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("Lidar + AI")

    def frame_2_button_event(self):
        self.select_frame_by_name("Interpolated Lidar")

    def frame_3_button_event(self):
        self.select_frame_by_name("Camera")


if __name__ == "__main__":
    # start mqtt client in a thread with threading
    asyncio.get_event_loop().close()
    app = App()
    threading.Thread(target=mqtt_thread, args=(app,)).start()
    app.mainloop()
