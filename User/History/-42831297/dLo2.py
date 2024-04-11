"""Run Inference using TensorRT.
Args
    --weights: onnx weights path or trt engine path; default yolov5s will download pretrained weights.
    --inputs: path to input video or image file. default people.mp4 will download demo video.
    --output: path to output video or image file. default out.mp4 (out.jpg if image file given in input )
"""
import argparse
import asyncio
import base64
import json
import os
import time
import numpy as np
import cv2
from cvu.detector.yolov5 import Yolov5 as Yolov5Trt
from vidsz.opencv import Reader
import pyudev
from detect import detect_mqtt_broker
from lidar_module import LIDAR
from lidar_module.TOF import Tof
from usb_stuff import USB_Dispatcher
import threading
from paho.mqtt.client import Client


yolo_lock = threading.Lock()
mqtt_client: Client
detections = []
angle_lower_limit_1 = 318
angle_upper_limit_1 = 383
angle_lower_limit_2 = 23
angle_upper_limit_2 = 47

ai_threshold = 1
lidar_threshold = 3


send_frames = False
usb_disp = USB_Dispatcher(pyudev.Context())

lidar_addr: list


def construct_mqtt_packet(
    distances,
    people,
    angle_lower_limit_1,
    angle_upper_limit_1,
    angle_lower_limit_2,
    angle_upper_limit_2,
    image,
):
    # people should be a list made of tuple(person_angle, distance)

    packet = {
        "limits": [angle_lower_limit_1, angle_upper_limit_1, angle_lower_limit_2, angle_upper_limit_2],
        "image": image,
        "points": distances,
        "people": people,
    }
    packet = {k: v for k, v in packet.items() if v is not None}

    return json.dumps(packet)


def get_yolo_frames():
    with yolo_lock:
        local_detections = detections.copy()
        detections.clear()
        return local_detections


def decode_yolo_dataframe(lidar, slices, dataframe):
    try:
        people_angles, image = dataframe
    except Exception as e:
        print(f"Error getting people angles: {e}")
        return None, None
    if people_angles is not None:
        people = []
        for angle, _ in people_angles:
            distance, _ = lidar.get_distance_at_angle(slices, angle)
            if distance is not None:
                people.append((angle, distance))
        return people, image


def detect_video(weight, input_video, auto_install=True, dtype="fp16"):
    while True:
        try:
            # load model
            model = Yolov5Trt(classes="coco", backend="tensorrt", weight=weight, auto_install=auto_install, dtype=dtype)

            # print(model._classes)
            reader = Reader(input_video)

            warmup = np.random.randint(0, 255, reader.read().shape).astype("float")
            for i in range(100):
                model(warmup)

            inference_time = 0
            c = 0
            start_fps = time.time()
            for frame in reader:
                frame = np.rot90(frame, 2)
                frame[:, frame.shape[1] // 2 :] = 0
                frame = frame.copy()

                # inference
                start = time.time()
                preds = model(frame)
                inference_time += time.time() - start

                c += 1
                if c % 60 == 0:
                    print(f"FPS: {c / (time.time() - start_fps):.02f}")
                    start_fps = time.time()
                    c = 0

                frame = frame[:, : frame.shape[1] // 2]
                # preds.draw(rotated)

                people_angles = []

                for pred in preds:
                    bbox = pred.bbox
                    x_center = float((bbox[0] + bbox[2])) / 2
                    # get x as fraction of image width
                    x_center = x_center / frame.shape[1]

                    total_fov = (angle_upper_limit_2 - angle_lower_limit_2) + (angle_upper_limit_1 - angle_lower_limit_1)
                    x_center *= total_fov
                    angle_ = x_center + angle_lower_limit_1
                    if angle_lower_limit_1 > angle_upper_limit_2 and angle_ > angle_upper_limit_1:
                        angle_ -= 360
                    # print(f"Time to draw: {time.time() - start_drawing:.04f} s")

                    if pred.confidence > 0.6 and pred.class_name == "person":
                        pred.draw(frame)
                        # print(f"Confidence: {pred.confidence}")

                        people_angles.append((angle_, bbox))

                        # draw center on image as circle at y height/2
                        # cv2.circle(frame, (int(x_center * frame.shape[1]), frame.shape[0] // 2), 5, (0, 0, 255), -1)

                with yolo_lock:
                    detections.clear()
                    detections.append((people_angles, frame))

            reader.release()
        except Exception as e:
            print("Exception in yolo: ", e)
            os.kill(os.getpid(), 9)
            continue


def mqtt(broker_ip):
    global mqtt_client
    while True:
        try:
            mqtt_client = Client(reconnect_on_failure=False)

            def on_message(client, userdata, message):
                print(f"Message received: {message.payload}")

                try:
                    json_data = json.loads(message.payload)
                except Exception as e:
                    print("Couldn't parse mqtt packet: ", e)
                    return

                # get the command
                if "send_frames" in json_data:
                    global send_frames
                    send_frames = json_data["send_frames"]
                    print(f"Send frames set to {send_frames}")
                elif (
                    "angle_lower_limit_1" in json_data
                    and "angle_upper_limit_1" in json_data
                    and "angle_lower_limit_2" in json_data
                    and "angle_upper_limit_2" in json_data
                ):
                    global angle_lower_limit_1, angle_upper_limit_1, angle_lower_limit_2, angle_upper_limit_2
                    angle_lower_limit_1 = json_data["angle_lower_limit_1"]
                    angle_upper_limit_1 = json_data["angle_upper_limit_1"]
                    angle_lower_limit_2 = json_data["angle_lower_limit_2"]
                    angle_upper_limit_2 = json_data["angle_upper_limit_2"]
                    print(f"Angle limits set to {angle_lower_limit_1}, {angle_upper_limit_1}, {angle_lower_limit_2}, {angle_upper_limit_2}")
                if "lidar_threshold" in json_data:
                    global lidar_threshold
                    lidar_threshold = json_data["lidar_threshold"]
                    print(f"Lidar threshold set to {lidar_threshold}")
                else:
                    print(json_data)

            def on_subscribe(client, userdata, mid, granted_qos):
                print(f"Subscribed: {str(mid)} {str(granted_qos)}")

            def on_unsubscribe(client, userdata, mid):
                print(f"Unsubscribed: {str(mid)}")

            def on_connect(client, userdata, flags, rc):
                print(f"Connected with result code {str(rc)}")
                client.subscribe("humidity/command", 1)

            mqtt_client.on_message = on_message
            mqtt_client.on_subscribe = on_subscribe
            mqtt_client.on_unsubscribe = on_unsubscribe
            mqtt_client.on_connect = on_connect
            if not broker_ip:
                print("No broker found")
            mqtt_client.connect(broker_ip, 1883, 60)
            mqtt_client.loop_forever()
        except Exception as e:
            print(f"Error in mqtt thread: {e}")
            time.sleep(1)


def commander():
    """ """
    try:
        lidar = LIDAR.Lidar(lidar_addr[0], "Lidar", slices=True, buffer_size=1)
        lidar.active(True)
        print("Lidar active")
        yolo_dataframes = []
        while True:
            time.sleep(0.03)
            distances = lidar.get_distances(pop=False)
            slices = lidar.get_slices(pop=False)
            yolo_dataframes = get_yolo_frames()
            people = None
            image = None

            last_yolo_dataframe = yolo_dataframes[-1]
            decode_yolo_dataframe(last_yolo_dataframe, slices, distances)

            if image is not None:
                try:
                    mqtt_client.publish(
                        "humidity/feed",
                        construct_mqtt_packet(
                            distances,
                            people,
                            angle_lower_limit_1,
                            angle_upper_limit_1,
                            angle_lower_limit_2,
                            angle_upper_limit_2,
                            base64.b64encode(image.tostring()).decode("utf-8") if send_frames and image is not None else None,
                        ),
                    )
                except Exception as e:
                    print(f"Error publishing: {e}")

            yolo_dataframes.clear()

    except Exception as e:
        # killiamo tutto il processo se il commander thread muore
        print(f"Error in commander thread: {e}")
        os.kill(os.getpid(), 9)


if __name__ == "__main__":
    # chiudiamo questo perché useremo soltanto l'event loop nel thread mqtt_thread
    asyncio.get_event_loop().close()

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.5.213", help="broker ip address")
    parser.add_argument("--weights", type=str, default="yolov5s", help="onnx weights path or trt engine path")
    parser.add_argument("--input", type=str, default="/dev/video0", help="path to input video or image file")
    parser.add_argument("--no-auto-install", action="store_true", help="Turn off auto install feature")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="set engine precision")
    opt = parser.parse_args()

    # image file
    input_ext = os.path.splitext(opt.input)[-1]

    lidar_addr = usb_disp.get_tty_by_devicename("Lidar")
    # tof2_addr = usb_disp.get_tty_by_devicename("TOF2")

    # thread con client mqtt per inviare dati via lan all'interfaccia grafica
    mqtt_thread = threading.Thread(target=mqtt, args=(opt.ip,))
    mqtt_thread.start()

    # thread per raccogliere i dati dai vari sensori e passarli al thread mqtt una volta  elaborati
    commander_thread = threading.Thread(target=commander)
    commander_thread.start()

    # thread di yolo lol
    yolo_thread = threading.Thread(target=detect_video, args=(opt.weights, opt.input, not opt.no_auto_install, opt.dtype.lower()))
    yolo_thread.start()
