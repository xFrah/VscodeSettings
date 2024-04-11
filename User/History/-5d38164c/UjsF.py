import argparse
import threading
import time

import cv2
import numpy
import serial
from matplotlib import pyplot as plt


def decode(line):
    # TODO: Stop creating lists here
    checksum = line[10020]
    summe = sum(line[:10020]) % 256
    if checksum != summe:
        return
    # stringa = f"{checksum:08b} {summe:08b}"
    image_ = line[20:10020]
    try:
        arr = numpy.frombuffer(image_, numpy.uint8)
        arr = arr.reshape(100, 100)
    except ValueError:
        return
    return arr


class Tof:
    def __init__(self, addr: list, name, usb_manager, buffer_size: int = 5, background_shape=(100, 100), matrix_path=None, baudrate=921600):
        self.addr = addr
        self.name = name
        self.usb_manager = usb_manager
        self.baudrate = baudrate
        # print("Tof >> Opening serial port: ", self.com)
        self.current_addr = 0
        self.serial_tof: serial.Serial = serial.Serial(self.addr[self.current_addr], self.baudrate, rtscts=True, dsrdtr=True)
        self.read_thread: threading.Thread = threading.Thread(target=self._read_data)
        self.buffer_size = buffer_size
        self.background_shape = background_shape
        self.background_padding = numpy.zeros((background_shape[0], background_shape[1], 4), numpy.uint8)
        self.background_padding_gray = numpy.zeros((background_shape[0], background_shape[1]), numpy.uint8)
        self.data = bytes()
        self.images = []
        self.reading = True
        self.colormap = plt.get_cmap("rainbow")
        # load matrix with numpy at path matrix_path
        self.M = numpy.load(matrix_path) if matrix_path else None
        self.M_inv = None
        # send at command to start streaming
        self.serial_tof.write(b"AT+DISP=3\r")
        self.serial_tof.write(b"AT+FPS=15\r")
        self.serial_tof.write(b"AT+BAUD=5\r")
        self.read_thread.start()

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
    angle_lower_limit_1 = 308
    angle_upper_limit_1 = 383
    angle_lower_limit_2 = 23
    angle_upper_limit_2 = 37

    ai_threshold = 1
    tof_threshold = 1


    broker_ip: str = None
    send_frames = False
    tof_roi = [0, 0, 100, 100]
    usb_disp = USB_Dispatcher(pyudev.Context())

    lidar_addr: list
    tof1_addr: list
    tof2_addr: list


    def construct_mqtt_packet(
        lidar,
        tof1,
        tof2,
        distances,
        people,
        angle_lower_limit_1,
        angle_upper_limit_1,
        angle_lower_limit_2,
        angle_upper_limit_2,
        image,
        dangers,
        # yolo_fps,
    ):
        # people should be a list made of tuple(person_angle, distance)

        packet = {
            "limits": [angle_lower_limit_1, angle_upper_limit_1, angle_lower_limit_2, angle_upper_limit_2],
        }
        # add image to packet if image is not None
        if image is not None:
            packet["image"] = image
        if distances is not None:
            packet["points"] = distances
        if people is not None:
            packet["people"] = people
        if tof1 is not None:
            packet["tof1_frame"] = tof1
        if tof2 is not None:
            packet["tof2_frame"] = tof2
        if dangers is not None:
            packet["dangers"] = dangers

        return json.dumps(packet)


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

                    start_drawing = time.time()
                    for pred in preds:
                        bbox = pred.bbox
                        x_center = float((bbox[0] + bbox[2])) / 2
                        # get x as fraction of image width
                        x_center = x_center / frame.shape[1]

                        total_fov = (angle_upper_limit_2 - angle_lower_limit_2) + (angle_upper_limit_1 - angle_lower_limit_1)
                        x_center *= total_fov
                        angle_ = x_center + angle_lower_limit_1
                        # print("Angle: ", angle_)
                        if angle_lower_limit_1 > angle_upper_limit_2 and angle_ > angle_upper_limit_1:
                            angle_ -= 360
                        # print(f"Time to draw: {time.time() - start_drawing:.04f} s")

                        if pred.confidence > 0.7:
                            pred.draw(frame)
                            # print(f"Confidence: {pred.confidence}")

                            people_angles.append((angle_, bbox))

                            # draw center on image as circle at y height/2
                            cv2.circle(frame, (int(x_center * frame.shape[1]), frame.shape[0] // 2), 5, (0, 0, 255), -1)

                    with yolo_lock:
                        detections.clear()
                        detections.append((people_angles, frame))

                    # cv2.imshow("frame", frame)
                    # cv2.waitKey(1)
                reader.release()
            except Exception as e:
                print("Exception in yolo: ", e)
                os.kill(os.getpid(), 9)
                continue


    def mqtt():
        global mqtt_client, broker_ip
        while True:
            try:
                detect_mqtt_broker()

                if not broker_ip:
                    print("No broker found")
                    broker_ip = "192.168.1.227"

                mqtt_client = Client()

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
                    elif "tof_roi" in json_data:
                        global tof_roi
                        tof_roi = json_data["tof_roi"]
                        print(f"ToF ROI set to {tof_roi}")

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
                mqtt_client.connect(broker_ip if broker_ip else "192.168.1.227", 1883, 60)
                mqtt_client.loop_forever()
            except Exception as e:
                print(f"Error in mqtt thread: {e}")
                time.sleep(1)


    def commander():
        try:
            lidar = LIDAR.Lidar(lidar_addr[0], "Lidar", usb_disp, slices=True, buffer_size=1)
            lidar.active(True)
            print("Lidar active")
            tof1 = Tof("TOF1", usb_disp, rotation=1)
            print("TOF1 active")
            tof2 = Tof("TOF2", usb_disp, rotation=-1)
            print("TOF2 active")
            start_time = time.time()

            c = 0
            local_detections = []
            while True:
                time.sleep(0.03)
                # get cluster of detection frames from yolo,
                # each one of those frames contains a list of slices in which the object was detected
                distances = lidar.get_distances(pop=False)
                slices = lidar.get_slices(pop=False)
                tof1_frame = tof1.get_image(pop=False, rotation=True)
                if tof1_frame is not None:
                    print(tof1_frame.shape)
                tof2_frame = tof2.get_image(pop=False, rotation=True)
                with yolo_lock:
                    if len(detections) != 0:
                        local_detections = detections.copy()
                        detections.clear()
                    else:
                        continue
                    # print(f"Detections: {len(local_detections)}")

                dangers = {
                    "lidar": False,
                    "tof": False,
                }
                if tof1_frame is not None:
                    tof1_frame = tof1_frame[tof_roi[1] : tof_roi[3], tof_roi[0] : tof_roi[2]]
                    # if at least a fourth of the pixels are of value >= threshold then there is a danger
                    if np.count_nonzero(tof1_frame >= 100) >= tof1_frame.size / 4:
                        dangers["tof"] = True
                if tof2_frame is not None:
                    tof2_frame = tof2_frame[tof_roi[1] : tof_roi[3], tof_roi[0] : tof_roi[2]]
                    # if at least a fourth of the pixels are of value >= threshold then there is a danger
                    if np.count_nonzero(tof2_frame >= tof_threshold) >= tof2_frame.size / 4:
                        dangers["tof"] = True

                if len(local_detections) > 0:
                    try:
                        people_angles, frame = local_detections[-1]
                    except Exception as e:
                        print(f"Error getting people angles: {e}")
                        people_angles = None
                        frame = None
                    if people_angles is not None:
                        try:
                            people = []
                            for angle, _ in people_angles:
                                d = lidar.get_distance_at_angle(slices, angle)
                                people.append((angle, d[0]))
                        except Exception as e:
                            print(f"Error getting people distances: {e}")
                            people = None
                        if people is not None:
                            for _, distance in people:
                                if distance <= ai_threshold:
                                    dangers["lidar"] = True
                else:
                    people = None
                    frame = None

                if frame is not None:
                    try:
                        mqtt_client.publish(
                            "humidity/feed",
                            construct_mqtt_packet(
                                lidar,
                                base64.b64encode(tof1_frame.tostring()).decode("utf-8") if tof1_frame is not None else None,
                                base64.b64encode(tof2_frame.tostring()).decode("utf-8") if tof2_frame is not None else None,
                                distances,
                                people if people is not None else None,
                                angle_lower_limit_1,
                                angle_upper_limit_1,
                                angle_lower_limit_2,
                                angle_upper_limit_2,
                                base64.b64encode(frame.tostring()).decode("utf-8") if send_frames and frame is not None else None,
                                dangers,
                            ),
                        )
                    except Exception as e:
                        print(f"Error publishing: {e}")

                local_detections.clear()

        except Exception as e:
            print(f"Error in commander thread: {e}")
            os.kill(os.getpid(), 9)


    if __name__ == "__main__":
        # close asyncio event loop
        asyncio.get_event_loop().close()

        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolov5s", help="onnx weights path or trt engine path")
        parser.add_argument("--input", type=str, default="/dev/video0", help="path to input video or image file")
        parser.add_argument("--no-auto-install", action="store_true", help="Turn off auto install feature")
        parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="set engine precision")
        opt = parser.parse_args()

        # image file
        input_ext = os.path.splitext(opt.input)[-1]

        lidar_addr = usb_disp.get_tty_by_devicename("Lidar")
        tof1_addr = usb_disp.get_tty_by_devicename("TOF1")
        tof2_addr = usb_disp.get_tty_by_devicename("TOF2")

        mqtt_thread = threading.Thread(target=mqtt)
        mqtt_thread.start()

        commander_thread = threading.Thread(target=commander)
        commander_thread.start()

        yolo_thread = threading.Thread(target=detect_video, args=(opt.weights, opt.input, not opt.no_auto_install, opt.dtype.lower()))
        yolo_thread.start()

        # detect_video(opt.weights, opt.input, not opt.no_auto_install, opt.dtype.lower())


    def _read_data(self):
        last = time.time()
        start = time.time()
        c = 0
        retries = 0
        while True:
            # get number of bytes that are ready to be read by serial
            to_add = self.serial_tof.read_all()
            if not to_add:
                time.sleep(0.02)
                if last < time.time() - 1:
                    # This check is necessary because the usb reset could be coming from the other tof.
                    if self.usb_manager.resetting:
                        last = time.time()
                        continue
                    retries += 1
                    if retries > 5:
                        self.close()
                        self.usb_manager.reset(self.name)
                        time.sleep(1)

                        self.reload_serial_connection()
                        print("Tof >> Too much retries, hard usb reset.")
                        retries = 0
                    else:
                        self.reload_serial_connection()
                        print(f"Tof-{self.addr} >> No data for 1 second, turning to {self.addr[self.current_addr]}")
                    last = time.time()
                continue
            last = time.time()
            self.data += to_add
            # print to add
            # print(f"Tof >> Read {len(to_add)} bytes, total: {len(self.data)}")
            if len(self.data) == 0 or len(self.data) < 24000:
                time.sleep(0.005)
                continue

            # start = self.data.find(b"\x00\xff\x20\x27")
            start = self.data.find(b"\x00\xff")
            if start == -1:
                print("Tof >> Header not found, len: ", str(len(self.data)))
                self.data = bytes()
            else:
                self.data = self.data[start:]
                # TODO: Replace this with del self.data[:start]
                # print(f"Removed {start} ")
                end = 10022
                if len(self.data) >= end:
                    data_to_decode = self.data[:end]
                    image = decode(data_to_decode)
                    if image is not None:
                        self.images.append(image)
                        c += 1
                        if c == 10:
                            # print(f"Tof >> FPS: {c / (time.time() - start)}")
                            start = time.time()
                            c = 0
                    self.images[: -self.buffer_size] = []

                    # check if self.data has index end + 1
                    self.data = self.data[end:]
                    # TODO: Replace this with del self.data[:end]
                    # print(f"Removed {end + 2} bytes")
            time.sleep(0.01)

    def get_distance_at_cv2_rectangle(self, x: int, y: int, w: int, h: int) -> tuple or None:
        if not isinstance(x, int) or not isinstance(y, int) or not isinstance(w, int) or not isinstance(h, int):
            return print("Tof >> Invalid type, get_distance_at_cv2_rectangle takes 4 ints as arguments.")
        image_ = self.get_image(pop=False)

        if image_ is None:
            return None

        warp = self.warp_image(image_, gray=True)

        # sort the distances in rectangle and get the first half
        # dist_list = sorted(warp[y : y + h, x : x + w].flatten())[: int(w * h / 2)]
        median_dist = numpy.median(warp[y : y + h, x : x + w])
        return median_dist, warp

    def get_next_addr(self):
        self.current_addr += 1
        if self.current_addr >= len(self.addr):
            self.current_addr = 0
        return self.addr[self.current_addr]

    def change_warp_matrix(self, src: numpy.ndarray, dst: numpy.ndarray):
        self.M = cv2.getPerspectiveTransform(src, dst)
        # now find the matrix that would get the image back to normal
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def normalize_image(self, frame: numpy.ndarray, gray=False) -> numpy.ndarray:
        if gray:
            new_bg = self.background_padding_gray.copy()
            new_bg[:100, :100] = frame
        else:
            new_bg = self.background_padding.copy()
            new_bg[:100, :100, :] = frame
        return new_bg

    def color_image(self, frame: numpy.ndarray) -> numpy.ndarray:
        frame = self.colormap(frame / 255)[:, :, :3] * 255
        frame = frame.astype(numpy.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    def get_image(self, pop: bool = True):
        # print("Tof >> Getting image, buffer size: ", len(self.images))
        return (list.pop if pop else list.__getitem__)(self.images, 0) if len(self.images) else None

    def warp_image(self, image, gray=False) -> numpy.ndarray:
        image = self.normalize_image(image, gray=gray)
        if self.M is None:
            return image
        return cv2.warpPerspective(image, self.M, self.background_shape[::-1])

    def fix_buffer_leak(self):
        if len(self.images) > self.buffer_size:
            self.images[: -self.buffer_size] = []
            print("Tof >> Buffer leak fixed, buffer size: ", len(self.images))

    def active(self, is_reading: bool):
        if not isinstance(is_reading, bool):
            return print("Tof >> Invalid type, active takes a bool as argument.")
        self.reading = is_reading

    def reload_serial_connection(self):
        self.serial_tof.close()
        self.serial_tof = serial.Serial(self.get_next_addr(), self.baudrate, rtscts=True, dsrdtr=True)

    def close(self):
        self.serial_tof.close()


image = None
image_buffer = []
c = 0
start = time.time()
if __name__ == "__main__":
    # parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="COM8", help="Serial port address")
    args = parser.parse_args()
    print("Tof >> Serial port address: ", args.addr)
    tof = Tof(addr=args.addr, buffer_size=30)
    while True:
        if tof.images:
            # print("Tof >> Image in buffer")
            # get numpy array from bytes
            image_t = tof.get_image()
            if image_t is None:
                print("Invalid image")
            else:
                image = image_t
                c += 1
                if c % 10 == 0:
                    print(f"FPS: {c / (time.time() - start)}")
                    c = 0
                    start = time.time()

        if image is not None:
            cv2.imshow("image", image)
            image_buffer.append(image)
            # wait one second for key press
            key = cv2.waitKey(1)
            if key == ord("q"):
                # save all images in buffer
                for i, img in enumerate(image_buffer):
                    cv2.imwrite(f"image_{i}.png", img)
                break
