import base64
import json
import math
import os
import time
import tkinter
import customtkinter
import threading
from PIL import Image
import numpy as np
import cv2
from tkterminal import Terminal
import asyncio
import asyncio_mqtt
from process_killer import kill_by_process_name
from matplotlib import pyplot as plt


last_frame = None


def compute_new_angles(first, second):
    if second < 23.0:
        second += 360

    a1 = first
    a4 = second

    mid = (second - a1) // 2
    a2 = a1 + mid + 1
    a3 = a2 - 1

    if a1 > second:
        a2 = 383
        a3 = 23
    print(f"New limits: {a1}° - {a2}° and {a3}° - {a4}°")
    # app.publish_new_limits(temp_values)


class StatusPanel:
    def __init__(self, master, labels: tuple = ("Status", "Temperature", "Humidity", "Pressure", "Altitude")):
        # make a frame for the status labels
        self.status_labels_frame = customtkinter.CTkFrame(master, corner_radius=10)

        self.colors = {
            "red": customtkinter.CTkImage(Image.open("Ellisse 3014.png")),
            "green": customtkinter.CTkImage(Image.open("Ellisse 3015.png")),
            "orange": customtkinter.CTkImage(Image.open("Ellisse 3016.png")),
            "grey": customtkinter.CTkImage(Image.open("Ellisse 3017.png")),
            "yellow": customtkinter.CTkImage(Image.open("Ellisse 3018.png")),
        }

        self.labels: dict[str, tuple[customtkinter.CTkLabel, customtkinter.CTkLabel, customtkinter.CTkLabel]] = {
            name: (
                customtkinter.CTkLabel(self.status_labels_frame, text=name + ":", font=customtkinter.CTkFont(size=12)),
                customtkinter.CTkLabel(self.status_labels_frame, text="N/A", font=customtkinter.CTkFont(size=12)),
                customtkinter.CTkLabel(self.status_labels_frame, text="", font=customtkinter.CTkFont(size=12), image=self.colors["grey"]),
            )
            for name in labels
        }

    def grid(self, row=0, column=0):
        self.status_labels_frame.grid(row=row, column=column, padx=20, pady=20, sticky="nsew")
        for i, (label, var, image) in enumerate(self.labels.values()):
            label.grid(row=i, column=0, sticky="w", padx=20, pady=(3 if i != 0 else 10, 0 if i != len(self.labels) - 1 else 10))
            var.grid(row=i, column=1, padx=20, pady=(3 if i != 0 else 10, 0 if i != len(self.labels) - 1 else 10))
            image.grid(row=i, column=2, sticky="e", padx=20, pady=(3 if i != 0 else 10, 0 if i != len(self.labels) - 1 else 10))
        # make column 2 stick to the right
        self.status_labels_frame.grid_columnconfigure(2, weight=1)

    def set_status(self, variable: str, status: str, color: str):
        self.labels[variable][1].configure(text=status)
        self.labels[variable][2].configure(image=self.colors[color])


class my_mqtt_client:
    def __init__(self, host: str = "localhost", topic: str = "humidity/feed", client_id: str = "MQTT_CLIENT_" + str(time.time())):
        self.client: asyncio_mqtt.Client
        self.stopped: bool = False
        self.frame_buffer: list[str] = []
        self.to_send_buffer: list[tuple[str, str]] = []
        self.to_send_buffer_lock: threading.Lock = threading.Lock()
        self.frame_buffer_lock: threading.Lock = threading.Lock()
        self.host: str = host
        self.topic: str = topic
        self.client_id: str = client_id
        self.connected: bool = False
        self.rx_buffer_size: int = 1
        self.thread: threading.Thread = threading.Thread(target=self._mqtt_thread)
        self.thread.start()

    def decode(self, payload) -> dict or None:
        try:
            return json.loads(payload)
        except Exception:
            return None

    def start(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def data_ready(self) -> bool:
        return len(self.frame_buffer) > 0

    def get_frame(self) -> str:
        with self.frame_buffer_lock:
            return self.frame_buffer.pop(0)

    def send_frame(self, topic: str, frame: str) -> None:
        with self.to_send_buffer_lock:
            self.to_send_buffer.append((topic, frame))

    def buffer_frame(self, frame) -> None:
        with self.frame_buffer_lock:
            self.frame_buffer.append(frame)
            del self.frame_buffer[: -self.rx_buffer_size]

    def _mqtt_thread(self):
        while True:

            async def receiver(client):
                async with client.messages() as messages:
                    await client.subscribe(self.topic)
                    await client.subscribe("humidity/command")
                    print("MQTT >> Subscribed.")
                    self.connected = True
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
                self.connected = False
            else:
                time.sleep(0.7)
                self.connected = False


def lidar_draw(packet, scaling_factor, closest_person, closest_object, lidar_threshold, object_threshold) -> np.ndarray:
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

        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), 5)

        # print("person at angle: ", person_angle, "distance: ", distance)

    # draw fov with two lines
    cv2.line(
        img,
        (500, 500),
        (
            int(500 + (50000 * scale) * math.cos(math.radians(angle_lower_limit_1))),
            int(500 + (50000 * scale) * math.sin(math.radians(angle_lower_limit_1))),
        ),
        (0, 255, 0),
        1,
    )

    cv2.line(
        img,
        (500, 500),
        (
            int(500 + (50000 * scale) * math.cos(math.radians(angle_upper_limit_2))),
            int(500 + (50000 * scale) * math.sin(math.radians(angle_upper_limit_2))),
        ),
        (0, 255, 0),
        1,
    )

    # display arch between those two lines with radius lidar_threshold
    cv2.ellipse(
        img,
        (500, 500),
        (int(lidar_threshold * scale), int(lidar_threshold * scale)),
        0,
        angle_lower_limit_1,
        angle_upper_limit_1,
        (0, 255, 0),
        1,
    )

    # display arch between those two lines with radius lidar_threshold
    cv2.ellipse(
        img,
        (500, 500),
        (int(lidar_threshold * scale), int(lidar_threshold * scale)),
        0,
        angle_lower_limit_2,
        angle_upper_limit_2,
        (0, 255, 0),
        1,
    )

    # draw threshold for object_threshold as a circle with radius object_threshold
    cv2.circle(img, (500, 500), int(object_threshold * scale), (255, 255, 0), 1)

    # draw line to closest person
    if closest_person is not None:
        x = 500 + (closest_person[1] * scale) * math.cos(math.radians(closest_person[0]))
        y = 500 + (closest_person[1] * scale) * math.sin(math.radians(closest_person[0]))

        cv2.line(img, (500, 500), (int(x), int(y)), (0, 255, 0), 2)

        # show little white label with distance on top of a little dark grey rectangle, like a tooltip
        # if possible the rectangle should blend with an alpha channel
        cv2.rectangle(img, (int(x) - 30, int(y) - 25), (int(x) + 85, int(y) - 60), (50, 50, 50), -1)
        cv2.putText(
            img,
            f"{closest_person[1] / 1000:.02f}m",
            (int(x) - 20, int(y) - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # draw line to closest object
    if closest_object is not None:
        x = 500 + (closest_object[1] * scale) * math.cos(math.radians(closest_object[0]))
        y = 500 + (closest_object[1] * scale) * math.sin(math.radians(closest_object[0]))

        # draw the line yellow
        cv2.line(img, (500, 500), (int(x), int(y)), (255, 255, 0), 2)

        # show little white label with distance on top of a little dark grey rectangle, like a tooltip
        # if possible the rectangle should blend with an alpha channel
        cv2.rectangle(img, (int(x) - 30, int(y) - 25), (int(x) + 85, int(y) - 60), (50, 50, 50), -1)
        cv2.putText(
            img,
            f"{closest_object[1] / 1000:.02f}m",
            (int(x) - 20, int(y) - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return img


colormap = plt.get_cmap("rainbow")


def color_image(frame: np.ndarray) -> np.ndarray:
    frame = colormap(frame / 255)[:, :, :3] * 255
    frame = frame.astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class App(customtkinter.CTk):
    def __init__(self, mqtt_client):
        super().__init__()

        self.mqtt_client: my_mqtt_client = mqtt_client
        self.lidar_scaling = 0.15
        self.image = None
        self.calibration = 0
        self.lidar_threshold = 1000
        self.object_threshold = 1000
        self.tof_roi = ((5, 5), (95, 70))
        self.send_frames = False

        self.title("")
        self.geometry("1000x800")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        self.logo_image = customtkinter.CTkImage(Image.open("leftlogo.png"), size=(70, 32))
        self.lidar_image = customtkinter.CTkImage(Image.open("lidar.png"), size=(25, 15))
        self.tof_image = customtkinter.CTkImage(Image.open("tof.png"), size=(25, 15))
        self.camera_image = customtkinter.CTkImage(Image.open("camera.png"), size=(25, 15))
        # put window icon
        self.iconphoto(True, self.iconify("leftlogo.png"))
        self.image = None
        self.image2 = None

        # ----------------------------- Navigation frame -----------------------------
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(
            self.navigation_frame,
            text="",
            compound="left",
            image=self.logo_image,
            font=customtkinter.CTkFont(size=20, weight="bold"),
        )
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.frame_1_button = customtkinter.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="  Full view",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            image=self.lidar_image,
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
            text="Broker",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=self.frame_2_button_event,
        )
        self.frame_2_button.grid(row=5, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="  Cam view",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            image=self.camera_image,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=self.frame_3_button_event,
        )
        self.frame_3_button.grid(row=3, column=0, sticky="ew")

        self.frame_4_button = customtkinter.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="  Tof view",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            image=self.tof_image,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=self.frame_4_button_event,
        )

        self.frame_4_button.grid(row=2, column=0, sticky="ew")

        # ----------------------------- Status frame -----------------------------

        self.status_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.status_frame.grid(row=0, column=2, sticky="nsew")
        self.status_frame.grid_rowconfigure(4, weight=1)

        def connect_callback():
            # get path to mosquitto executable, in program files
            mosquitto_path = '"' + os.path.join(os.environ["PROGRAMFILES"], "mosquitto", "mosquitto.exe") + '"'
            conf = '"' + os.path.join(os.environ["PROGRAMFILES"], "mosquitto", "mosquitto.conf") + '"'
            self.term.run_command(mosquitto_path + " -c " + conf)
            self.mqtt_client.start()

        def disconnect_callback():
            self.mqtt_client.stop()
            kill_by_process_name("mosquitto.exe")

        def reboot_callback():
            self.mqtt_client.send_frame("humidity/command", json.dumps({"reboot": True}))

        self.status_panel = StatusPanel(
            self.status_frame, labels=("Broker", "People detected", "Closest person", "Closest object", "Tof distance")
        )
        self.status_panel.grid()

        # create frame for grid of buttons
        self.status_buttons_frame = customtkinter.CTkFrame(self.status_frame, corner_radius=0, fg_color="transparent")

        # create a 4x4 grid of buttons named "Connect", "Disconnect", "Reboot", "Restart script" within the frame self.status_buttons_frame
        self.status_button_connect = customtkinter.CTkButton(
            self.status_buttons_frame,
            corner_radius=10,
            height=60,
            width=150,
            border_spacing=10,
            text="Connect",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            fg_color="gray20",
            command=connect_callback,
        )
        self.status_button_connect.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 5))

        self.status_button_disconnect = customtkinter.CTkButton(
            self.status_buttons_frame,
            corner_radius=10,
            height=60,
            width=150,
            border_spacing=10,
            text="Disconnect",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            fg_color="gray20",
            command=disconnect_callback,
        )

        self.status_button_disconnect.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 5))

        self.status_button_reboot = customtkinter.CTkButton(
            self.status_buttons_frame,
            corner_radius=10,
            width=150,
            height=60,
            border_spacing=10,
            text="Reboot",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            fg_color="gray20",
            # command=self.status_button_reboot_event,
        )

        self.status_button_reboot.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=(5, 0))

        self.status_button_restart_script = customtkinter.CTkButton(
            self.status_buttons_frame,
            corner_radius=10,
            width=150,
            height=60,
            border_spacing=10,
            text="Restart script",
            text_color=("gray10", "gray90"),
            fg_color="gray20",
            hover_color=("gray70", "gray30"),
            # command=self.status_button_restart_script_event,
        )

        self.status_button_restart_script.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))

        # self.status_buttons_frame.grid(row=2, column=0, padx=20)
        # display at the bottom
        self.status_buttons_frame.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 20))

        # ----------------------------- Frame 1 -----------------------------

        self.lidar_page = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.lidar_page.grid_columnconfigure(0, weight=1)

        self.lidar_feed = customtkinter.CTkLabel(self.lidar_page, text="", image=self.image)
        self.lidar_feed.grid(pady=(20, 0), padx=20)

        def callback_scaling(value):
            self.scaling_slider_label_value.configure(text=f"{int(value)/100:.02f}x")
            self.lidar_scaling = int(value) / 100

        def callback_threshold(value):
            self.threshold_slider_label_value.configure(text=f"{int(value)/1000:.02f}m")
            self.lidar_threshold = int(value)
            self.mqtt_client.send_frame("humidity/command", json.dumps({"lidar_threshold": self.lidar_threshold}))

        # create frame for sliders
        self.scaling_slider_frame = customtkinter.CTkFrame(self.lidar_page, corner_radius=12, width=400)
        self.scaling_slider_frame.grid(row=1, column=0, pady=15)

        self.scaling_slider_frame_label = customtkinter.CTkLabel(
            self.scaling_slider_frame,
            text="Scaling:",
            font=customtkinter.CTkFont(size=14, weight="bold"),
        )

        self.threshold_slider_frame = customtkinter.CTkFrame(self.lidar_page, corner_radius=12, width=400)
        self.threshold_slider_frame.grid(row=2, column=0)

        # enform

        self.threshold_slider_frame_label = customtkinter.CTkLabel(
            self.threshold_slider_frame,
            text="Thresholds:",
            font=customtkinter.CTkFont(size=14, weight="bold"),
        )

        self.threshold_slider_frame_label.grid(column=1, row=0, pady=(10, 0))

        # label the slider with a text
        self.threshold_slider_label = customtkinter.CTkLabel(
            self.threshold_slider_frame,
            text="People:",
            font=customtkinter.CTkFont(size=13),
        )

        self.threshold_slider_label.grid(column=0, row=1, padx=15, pady=(10, 25), sticky="w")

        # label the slider with a text
        self.threshold_slider_label_value = customtkinter.CTkLabel(
            self.threshold_slider_frame,
            text=f"{self.lidar_threshold/1000:.02f}m",
            font=customtkinter.CTkFont(size=13),
        )

        self.threshold_slider_label_value.grid(column=2, row=1, padx=15, pady=(10, 25))

        self.threshold_slider = customtkinter.CTkSlider(
            self.threshold_slider_frame,
            from_=300,
            to=20000,
            width=300,
            command=callback_threshold,
        )

        self.threshold_slider.grid(column=1, row=1, padx=15, pady=(10, 25))

        # ------------------------------
        self.threshold_2_slider_label = customtkinter.CTkLabel(
            self.threshold_slider_frame,
            text="Objects:",
            font=customtkinter.CTkFont(size=13),
        )

        self.threshold_2_slider_label.grid(column=0, row=2, padx=15, pady=(0, 30), sticky="w")

        # label the slider with a text
        self.threshold_2_slider_label_value = customtkinter.CTkLabel(
            self.threshold_slider_frame,
            text=f"{self.object_threshold/1000:.02f}m",
            font=customtkinter.CTkFont(size=13),
        )

        self.threshold_2_slider_label_value.grid(column=2, row=2, padx=15, pady=(0, 30))

        def callback_threshold_2(value):
            self.threshold_2_slider_label_value.configure(text=f"{int(value)/1000:.02f}m")
            self.object_threshold = int(value)

        self.threshold_2_slider = customtkinter.CTkSlider(
            self.threshold_slider_frame,
            from_=300,
            to=20000,
            width=300,
            command=callback_threshold_2,
        )

        self.threshold_2_slider.grid(column=1, row=2, padx=15, pady=(0, 30))

        # set the default value
        self.threshold_2_slider.set(self.object_threshold)

        # ------------------------------

        self.scaling_slider_frame_label.grid(column=1, row=0, pady=(10, 0))

        # label the slider with a text
        self.scaling_slider_label = customtkinter.CTkLabel(
            self.scaling_slider_frame,
            text="Threshold:",
            font=customtkinter.CTkFont(size=13),
        )
        self.scaling_slider_label.grid(column=0, row=1, padx=(15, 30), pady=(10, 30), sticky="w")

        # label the slider with a text
        self.scaling_slider_label_value = customtkinter.CTkLabel(
            self.scaling_slider_frame,
            text="0.15x",
            font=customtkinter.CTkFont(size=13),
        )
        self.scaling_slider_label_value.grid(column=2, row=1, padx=(30, 15), pady=(10, 30))

        self.scaling_slider = customtkinter.CTkSlider(
            master=self.scaling_slider_frame, from_=1, to=100, width=300, command=callback_scaling
        )
        self.scaling_slider.grid(column=1, row=1, pady=(10, 30))
        self.scaling_slider.set(15)

        self.threshold_slider.set(self.lidar_threshold)

        # upon clicking the button, two other sliders will be shown in the same frame, then they will be hidden again when the button is clicked again
        def callback_calibration():
            self.calibration = not self.calibration
            if self.calibration == 1:
                self.slider_2_label.grid()
                self.slider_2.grid(pady=10)
                self.slider_3_label.grid()
                self.slider_3.grid(pady=10)
            else:
                self.slider_2.grid_remove()
                self.slider_3.grid_remove()
                self.slider_2_label.grid_remove()
                self.slider_3_label.grid_remove()

        def callback_calibration_slider(value):
            a1, a2 = self.slider_2.get(), self.slider_3.get()
            compute_new_angles(a1, a2 + a1)

        def callback_camera_toggle():
            self.send_frames = not self.send_frames
            self.mqtt_client.send_frame("humidity/command", json.dumps({"send_frames": self.send_frames}))

        self.slider_2 = customtkinter.CTkSlider(master=self.lidar_page, from_=0, to=360, width=400, command=callback_calibration_slider)

        self.slider_2_label = customtkinter.CTkLabel(
            self.lidar_page,
            text="Lower angle",
        )
        self.slider_3 = customtkinter.CTkSlider(master=self.lidar_page, from_=90, to=180, width=400, command=callback_calibration_slider)

        self.slider_3_label = customtkinter.CTkLabel(
            self.lidar_page,
            text="Upper angle",
        )

        # ----------------------------- Frame 2 -----------------------------

        # create second frame
        self.mosquitto_page = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        kill_by_process_name("mosquitto.exe")

        self.term = Terminal(self.mosquitto_page, background="black", height=20, width=100, font=("Consolas", 25), foreground="white")
        self.term.shell = True

        def bind_term():
            self.term.bind("<<Cut>>", lambda e: "break", False)
            self.term.bind("<<Paste>>", lambda e: "break", False)

            self.term.bind("<Return>", lambda e: "break", False)
            self.term.bind("<KeyPress>", lambda e: "break", False)
            self.term.bind("<KeyPress>", lambda e: "break", False)
            self.term.bind("<BackSpace>", lambda e: "break", False)
            self.term.bind("<Button-1>", lambda e: "break", False)
            self.term.bind("<ButtonRelease-1>", lambda e: "break", False)
            self.term.bind("<Button-3>", lambda e: "break", False)
            self.term.bind("<ButtonRelease-3>", lambda e: "break", False)

            self.term.bind("<Command-k>", lambda e: "break", False)
            self.term.bind("<Control-c>", lambda e: "break", False)

        bind_term()

        self.term.basename = "mqtt$"

        self.term.grid(row=0, column=0, padx=20, pady=(20, 0))

        self.button_frame = customtkinter.CTkFrame(master=self.mosquitto_page)
        self.button_frame.grid(row=1, column=0)

        self.mosquitto_page.grid_columnconfigure(0, weight=1)

        # create button below the terminal
        self.connect_button = customtkinter.CTkButton(
            self.button_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Connect",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            command=connect_callback,
        )
        self.disconnect_button = customtkinter.CTkButton(
            self.button_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Disconnect",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            command=disconnect_callback,
        )

        self.connect_button.grid(row=1, column=0)
        self.disconnect_button.grid(row=1, column=1)

        # ----------------------------- Frame 3 -----------------------------

        # create third frame
        self.camera_page = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        self.cornice_camera = customtkinter.CTkFrame(master=self.camera_page)
        self.cornice_camera.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="n", ipadx=40)

        self.camera_feed = customtkinter.CTkLabel(self.cornice_camera, text="", image=self.image2)
        self.camera_feed.grid(pady=(20, 0), padx=40, sticky="w")

        # add button with callback callback_camera_toggle
        self.toggle_camera_button = customtkinter.CTkButton(
            self.camera_page,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Toggle camera",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            command=callback_camera_toggle,
        )
        self.toggle_camera_button.grid(row=1, column=0, pady=(20, 0))
        self.camera_page.grid_columnconfigure(0, weight=1)

        connect_callback()

        # ----------------------------- Frame 4 -----------------------------

        # create fourth frame
        self.tof_page = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.lidar_page.grid_columnconfigure(0, weight=1)

        self.tof1_feed = customtkinter.CTkLabel(master=self.tof_page, text="")
        self.tof1_feed.grid(row=0, column=1, padx=(0, 20), pady=(20, 0), sticky="n", ipadx=40)

        self.tof2_feed = customtkinter.CTkLabel(master=self.tof_page, text="")
        self.tof2_feed.grid(row=0, column=0, padx=(20, 0), pady=(20, 0), sticky="n", ipadx=40)

        def update_connection_status():
            self.status_panel.set_status(
                "Broker",
                "Connected" if self.mqtt_client.connected else "Disconnected",
                "green" if self.mqtt_client.connected else "red",
            )
            print("Connection status updated")

        def update_window_via_mqtt():
            start = time.time()
            c = 0
            old_connected = False
            update_connection_status()
            people_count_buffer = []
            lidar_alarm_buffer = []
            tof_alarm_buffer = []
            while True:
                time.sleep(0.01)
                if self.mqtt_client.connected != old_connected:
                    old_connected = self.mqtt_client.connected
                    update_connection_status()
                if self.mqtt_client.data_ready():
                    frame = self.mqtt_client.get_frame()
                    frame = self.mqtt_client.decode(frame)
                    if not frame:
                        continue

                    if "points" in frame and "people" in frame and "limits" in frame:
                        # people count blending
                        people_count_buffer.append(len(frame["people"]))
                        if len(people_count_buffer) > 25:
                            people_count_buffer.pop(0)
                        # get number that appears the most in the buffer
                        people_count = max(set(people_count_buffer), key=people_count_buffer.count)
                        self.status_panel.set_status("People detected", f"{people_count}", "green")

                        closest_person = None
                        closest_object = None
                        if len(frame["people"]) > 0:
                            closest_person = min(frame["people"], key=lambda x: x[1])
                            # lidar blending
                            lidar_alarm_buffer.append(closest_person[1] < self.lidar_threshold)
                            if len(lidar_alarm_buffer) > 25:
                                lidar_alarm_buffer.pop(0)
                            # get number that appears the most in the buffer
                            lidar_alarm = max(set(lidar_alarm_buffer), key=lidar_alarm_buffer.count)
                            self.status_panel.set_status(
                                "Closest person",
                                f"{closest_person[1] / 1000:.02f} m",
                                "red" if lidar_alarm else "green",
                            )
                        else:
                            self.status_panel.set_status("Closest person", "N/A", "green")
                        if len(frame["points"]) > 0:
                            closest_object = min([x for x in frame["points"] if x[1] > 50], key=lambda x: x[1])
                            self.status_panel.set_status(
                                "Closest object", f"{closest_object[1]/1000:.02f} m", "red" if closest_object[1] < 5000 else "green"
                            )
                        try:
                            image = Image.fromarray(
                                lidar_draw(frame, self.lidar_scaling, closest_person, closest_object, self.lidar_threshold, self.object_threshold)
                            )
                        except Exception as e:
                            print(e)
                            continue
                        try:
                            self.update_lidar_image(image)
                        except Exception as e:
                            print(e)
                    if "image" in frame:
                        try:
                            image2 = np.fromstring(base64.b64decode(frame["image"]), dtype=np.uint8).reshape(240, 320, 3)
                            self.update_camera_image(Image.fromarray(image2))
                        except Exception as e:
                            print(e)

                    tof1_mean = None
                    tof2_mean = None

                    if "tof1_frame" in frame:
                        try:
                            image3 = np.fromstring(base64.b64decode(frame["tof1_frame"]), dtype=np.uint8).reshape(100, 100)
                            # get median of the portion of the image that is inside the roi
                            roi = image3[self.tof_roi[0][1] : self.tof_roi[1][1], self.tof_roi[0][0] : self.tof_roi[1][0]]
                            # get 1/6 closest points of the roi
                            roi = roi.flatten()
                            roi.sort()
                            roi = roi[: int(len(roi) / 6)]
                            tof1_mean = np.median(roi)
                            # print(tof1_mean)
                            # color_image_ = color_image(image3)
                            # convert from grayscale to rgb

                            # color_image_ = cv2.cvtColor(image3, cv2.COLOR_GRAY2RGB)
                            color_image_ = color_image(image3)
                            # draw self.tof_roi as a rectangle on the image, self.tof_roi is a list of 2 points, up left and down right
                            cv2.rectangle(
                                color_image_,
                                (self.tof_roi[0][0], self.tof_roi[0][1]),
                                (self.tof_roi[1][0], self.tof_roi[1][1]),
                                (255, 255, 255),
                                1,
                            )
                            self.update_tof1_image(Image.fromarray(color_image_))
                        except Exception as e:
                            print(e)

                    if "tof2_frame" in frame:
                        try:
                            image4 = np.fromstring(base64.b64decode(frame["tof2_frame"]), dtype=np.uint8).reshape(100, 100)
                            roi = image4[self.tof_roi[0][1] : self.tof_roi[1][1], self.tof_roi[0][0] : self.tof_roi[1][0]]
                            roi = roi.flatten()
                            roi.sort()
                            roi = roi[: int(len(roi) / 6)]
                            tof2_mean = np.median(roi)
                            self.update_tof2_image(Image.fromarray(color_image(image4)))
                        except Exception as e:
                            print(e)
                    else:
                        print("No tof")

                    asd = []
                    if tof1_mean is not None:
                        asd.append(tof1_mean)
                    if tof2_mean is not None:
                        asd.append(tof2_mean)
                    if len(asd) > 0:
                        minimum = (min(asd) / 100) * 1000
                        if minimum < self.lidar_threshold:
                            tof_alarm_buffer += [True] * 2
                        else:
                            tof_alarm_buffer.append(False)
                        if len(tof_alarm_buffer) > 15:
                            tof_alarm_buffer.pop(0)
                        # get number that appears the most in the buffer
                        tof_alarm = max(set(tof_alarm_buffer), key=tof_alarm_buffer.count)
                        self.status_panel.set_status("Tof distance", f"{minimum/1000:.02f}m", "red" if tof_alarm else "green")
                    else:
                        if len(tof_alarm_buffer) > 0:
                            del tof_alarm_buffer[0]
                        if len(tof_alarm_buffer) == 0:
                            self.status_panel.set_status("Tof distance", "N/A", "grey")

                    c += 1
                    if c == 30:
                        print(f"FPS: {30 / (time.time() - start):.02f}")
                        start = time.time()
                        c = 0

        threading.Thread(target=update_window_via_mqtt).start()

        # select default frame
        self.select_frame_by_name("Full view")

    def update_lidar_image(self, image: Image.Image):
        # update the image in the home frame, fix the flickering
        image1 = customtkinter.CTkImage(image, size=(430, 430))
        self.lidar_feed.configure(image=image1)
        self.image = image1

    def update_camera_image(self, image: Image.Image):
        # update the image in the home frame, fix the flickering
        image1 = customtkinter.CTkImage(image, size=(320, 240))
        self.camera_feed.configure(image=image1)
        self.image2 = image1

    def update_tof1_image(self, image: Image.Image):
        # update the image in the home frame, fix the flickering
        image1 = customtkinter.CTkImage(image, size=(200, 200))
        self.tof1_feed.configure(image=image1)
        self.image3 = image1

    def update_tof2_image(self, image: Image.Image):
        # update the image in the home frame, fix the flickering
        image1 = customtkinter.CTkImage(image, size=(200, 200))
        self.tof2_feed.configure(image=image1)
        self.image4 = image1

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.frame_1_button.configure(fg_color="cornflower blue" if name == "Full view" else "transparent")
        self.frame_2_button.configure(fg_color="cornflower blue" if name == "Broker" else "transparent")
        self.frame_3_button.configure(fg_color="cornflower blue" if name == "Camera view" else "transparent")
        self.frame_4_button.configure(fg_color="cornflower blue" if name == "Tof view" else "transparent")

        # show selected frame
        if name == "Full view":
            self.lidar_page.grid(row=0, column=1, sticky="nsew")
        else:
            self.lidar_page.grid_forget()
        if name == "Broker":
            self.mosquitto_page.grid(row=0, column=1, sticky="nsew")
        else:
            self.mosquitto_page.grid_forget()
        if name == "Camera view":
            self.camera_page.grid(row=0, column=1, sticky="nsew")
        else:
            self.camera_page.grid_forget()
        if name == "Tof view":
            self.tof_page.grid(row=0, column=1, sticky="nsew")
        else:
            self.tof_page.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("Full view")

    def frame_2_button_event(self):
        self.select_frame_by_name("Broker")

    def frame_3_button_event(self):
        self.select_frame_by_name("Camera view")

    def frame_4_button_event(self):
        self.select_frame_by_name("Tof view")

    def publish_new_limits(self, dictionary):
        print("MQTT >> Publishing new limits: ", str(dictionary))
        self.mqtt_client.send_frame("humidity/command", json.dumps(dictionary))


if __name__ == "__main__":
    # start mqtt client in a thread with threading
    asyncio.get_event_loop().close()
    mqtt_client = my_mqtt_client()
    app = App(mqtt_client)
    mqtt_client.start()
    app.mainloop()
