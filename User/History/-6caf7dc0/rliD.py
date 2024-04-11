# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import base64
import json
import math
import os
import platform
import sys
import threading
import pyudev
import time
from pathlib import Path
from threading import Lock
import socket
import socketserver
from usb_stuff import USB_Dispatcher

import numpy as np
import torch

from lidar_module import LIDAR
from lidar_module.TOF import Tof

from paho.mqtt import client as mqtt_client

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


yolo_lock = Lock()
detections = []
angle_lower_limit_1 = 308
angle_upper_limit_1 = 383
angle_lower_limit_2 = 23
angle_upper_limit_2 = 37

broker_ip: str = None
send_frames = False
usb_disp = USB_Dispatcher(pyudev.Context())

lidar_addr: list
tof1_addr: list
tof2_addr: list


def scan_port(ip, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)  # Set socket timeout to 100 ms
        result = sock.connect_ex((ip, port))
        if result == 0:
            print("Found MQTT broker at:", ip)
            global broker_ip
            broker_ip = ip
        sock.close()
    except:
        pass


def detect_mqtt_broker():
    # Set the IP range to scan
    ip_range = "192.168.1."

    # Define the port to scan for
    port = 1883

    # Create a list of thread objects to scan the IP addresses in parallel
    threads = [threading.Thread(target=scan_port, args=(ip_range + str(i), port)) for i in range(1, 255)]

    # Start the threads and wait for them to finish
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if not broker_ip:
        print("No MQTT broker found")


def construct_mqtt_packet(
    lidar,
    tof1,
    tof2,
    distances,
    slices,
    people_angles,
    angle_lower_limit_1,
    angle_upper_limit_1,
    angle_lower_limit_2,
    angle_upper_limit_2,
    image,
    # yolo_fps,
):
    # people should be a list made of tuple(person_angle, distance)

    packet = {
        "points": distances,
        "people": [(angle, lidar.get_distance_at_angle(slices, angle)[0]) for angle, _ in people_angles]
        if people_angles and slices
        else [],
        "limits": [angle_lower_limit_1, angle_upper_limit_1, angle_lower_limit_2, angle_upper_limit_2],
        "image": image,
        # "yolo_fps": yolo_fps,
        # "lidar_fps": lidar.fps,
        # "tof1_fps": tof1.fps,
        # "tof2_fps": tof2.fps,
        # could add actual tof data
    }

    return json.dumps(packet)


@smart_inference_mode()
def yolo_run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    counter = 0
    start_time = time.time()
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    view_img = check_imshow(warn=True)
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    threading.Thread(target=commander, args=()).start()

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            s += "%gx%g " % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            people_angles = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # find the center of the bounding box
                    x_center = float((xyxy[0] + xyxy[2])) / 2
                    # get x in percentage of the image
                    x_center = x_center / (im0.shape[1] / 2)
                    # print("Percentage: ", x_center)
                    # print("Bounding box: ", xyxy)

                    total_fov = (angle_upper_limit_2 - angle_lower_limit_2) + (angle_upper_limit_1 - angle_lower_limit_1)
                    x_center *= total_fov
                    angle_ = x_center + angle_lower_limit_1
                    # print("Angle: ", angle_)
                    if angle_lower_limit_1 > angle_upper_limit_2 and angle_ > angle_upper_limit_1:
                        angle_ -= 360
                        # print("Angle_2: ", angle_ + 360)

                    people_angles.append((angle_, xyxy))

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

            with yolo_lock:
                detections.clear()
                detections.append((people_angles, annotator.result()))
            # print fps

            if counter == 10:
                print(f"FPS: {10 / (time.time() - start_time)}")
                start_time = time.time()
                counter = 0
            counter += 1

            # Stream results
            im0 = annotator.result()
            # if view_img:
            # if platform.system() == "Linux" and p not in windows:
            # windows.append(p)
            # cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            # cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            # cv2.imshow(str(p), im0)
            # cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_false", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", default="0", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def polisher(detections, detections_polished, slice_to_r_d):
    for detection_frame in detections:
        for slice_index, rect, distance in detection_frame:
            detections_polished[slice_index] = detections_polished.get(slice_index, 0) + 1
            slice_to_r_d[slice_index] = slice_to_r_d.get(slice_index, []) + [(rect, distance)]

        # age the detections to remove them if they are no longer in the picture
        for slice_index in detections_polished.copy():
            if slice_index not in (e[0] for e in detection_frame):
                detections_polished[slice_index] -= 1
            elif detections_polished[slice_index] >= 15:
                detections_polished[slice_index] = 15
                len_elem = len(slice_to_r_d[slice_index])
                if len_elem > 15:
                    del slice_to_r_d[slice_index][: len_elem - 15]
            if detections_polished[slice_index] <= 0:
                del detections_polished[slice_index]
                del slice_to_r_d[slice_index]


def commander():
    img_shape = (480, 1280)
    detect_mqtt_broker()
    lidar = LIDAR.Lidar(lidar_addr[0], "Lidar", usb_disp, slices=True, buffer_size=1)
    lidar.active(True)
    print("Lidar active")
    time.sleep(1)
    tof1 = Tof(tof1_addr, "TOF1", usb_disp, background_shape=img_shape, matrix_path="COM8-M.npy")
    print("TOF1 active")
    time.sleep(1)
    tof2 = Tof(tof2_addr, "TOF2", usb_disp, background_shape=img_shape, matrix_path="COM12-M.npy")
    print("TOF2 active")
    start_time = time.time()
    # create mqtt client
    client = mqtt_client.Client()

    def on_message(client, userdata, message):
        print(f"Message received: {message.payload}")
        # get json from message
        json_data = json.loads(message.payload)
        # get the command
        if "send_frames" in json_data:
            global send_frames
            send_frames = json_data["send_frames"]
            print(f"Send frames set to {send_frames}")

    client.on_message = on_message
    if not broker_ip:
        print("No broker found")
    client.connect(broker_ip if broker_ip else "192.168.1.227", 1883, 60)
    client.subscribe("humidity/command")
    print("Connected to MQTT?")
    client.loop_start()
    c = 0
    while True:
        time.sleep(0.01)
        # get cluster of detection frames from yolo,
        # each one of those frames contains a list of slices in which the object was detected
        distances = lidar.get_distances(pop=False)
        slices = lidar.get_slices(pop=False)
        tof1_frame = tof1.get_image(pop=False)
        tof2_frame = tof2.get_image(pop=False)
        with yolo_lock:
            if len(detections) == 0:
                continue
            local_detections = detections.copy()
            detections.clear()
            # print(f"Detections: {len(local_detections)}")

        for people_angles, frame in local_detections:
            client.publish(
                "humidity/feed",
                construct_mqtt_packet(
                    lidar,
                    tof1_frame,
                    tof2_frame,
                    distances,
                    slices,
                    people_angles,
                    angle_lower_limit_1,
                    angle_upper_limit_1,
                    angle_lower_limit_2,
                    angle_upper_limit_2,
                    base64.b64encode(frame.tostring()).decode("utf-8") if send_frames else None,
                ),
            )
            c += 1
            if c % 10 == 0:
                # print fps
                print("fps", 10 / (time.time() - start_time))
                start_time = time.time()
                c = 0


def main():
    global lidar_addr, tof1_addr, tof2_addr
    lidar_addr = usb_disp.get_tty_by_devicename("Lidar")
    tof1_addr = usb_disp.get_tty_by_devicename("TOF1")
    tof2_addr = usb_disp.get_tty_by_devicename("TOF2")

    opt = parse_opt()
    check_requirements(exclude=("tensorboard", "thop"))
    yolo_run(**vars(opt))


if __name__ == "__main__":
    main()
