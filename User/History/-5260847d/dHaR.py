from datetime import datetime
import json
import math
from helpers import ROI, delete_folder
import torch
import os
import cv2
import sys
import time
import change_detection
import argparse

from ultralytics import YOLO

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)

from coco_utils import COCO_test_helper
import numpy as np

normal_cut_time: int = 10
working: bool = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorbike ",
    "aeroplane ",
    "bus ",
    "train",
    "truck ",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign ",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog ",
    "horse ",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra ",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife ",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza ",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet ",
    "tvmonitor",
    "laptop	",
    "mouse	",
    "remote ",
    "keyboard ",
    "cell phone",
    "microwave ",
    "oven ",
    "toaster",
    "sink",
    "refrigerator ",
    "book",
    "clock",
    "vase",
    "scissors ",
    "teddy bear ",
    "hair drier",
    "toothbrush ",
)

coco_id_list = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


from rknnlite.api import RKNNLite


class VideoSaver:
    def __init__(self, fps: float, frame_size: tuple):
        self.video_output: cv2.VideoWriter = None
        self.fps = fps
        self.frame_size = frame_size

    def write(self, img):
        try:
            if self.video_output is None:
                self.cut_video()
            self.video_output.write(img)
        except Exception as e:
            print(e)

    def release(self):
        try:
            self.video_output.release()
        except Exception as e:
            pass

    def cut_video(self):
        self.release()
        self.start_time = time.time()
        self.start_time_text = datetime.now().isoformat()
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_text = os.path.join(args.cache_path, f"v{self.start_time:.2f}.mp4")
        self.video_output = cv2.VideoWriter(self.video_text, self.codec, self.fps, self.frame_size)

    def __del__(self):
        try:
            self.video_output.release()
        except Exception as e:
            pass


class RKNN_model_container:
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNNLite()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print("--> Init runtime environment")
        if target == None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)
        print("done")

        self.rknn = rknn

    def run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)

        return result


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    # Distribution Focal Loss (DFL)
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, "{0} {1:.2f}".format(CLASSES[cl], score), (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith(".pt") or model_path.endswith(".torchscript"):
        platform = "pytorch"
        from py_utils.pytorch_executor import Torch_model_container

        model = Torch_model_container(args.model_path)
    elif model_path.endswith(".rknn"):
        platform = "rknn"
        # from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith("onnx"):
        platform = "onnx"
        from py_utils.onnx_executor import ONNX_model_container

        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print("Model-{} is {} model, starting val".format(model_path, platform))
    return model, platform

def write_file(filtered_boxes, start_time_text, start_time) -> None:
    empty = len(filtered_boxes) == 0
    features_elem = {
        "empty": empty,
        "L1": True,
        "L2": False,
        "L3": False,
        "processing": False,
        "ts_init": start_time_text,
        "ts_final": datetime.now().isoformat(),
        "features": filtered_boxes,
    }

    json_text = os.path.join(args.cache_path, f"v{start_time:.2f}.json")
    if not os.path.exists(json_text):
        with open(json_text, "w") as json_file:
            json.dump(features_elem, json_file)

    print("File written")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model_path", type=str, default="yolo-Weights/yolov8.rknn", help="model path, could be .pt or .rknn file")
    parser.add_argument("--target", type=str, default=None, help="target RKNPU platform")
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    parser.add_argument("--video_path", help="path to video file", default="media/Sedili.mp4")
    parser.add_argument("--roi_path", help="path to roi file", default="ROI/2/ROI.csv")
    parser.add_argument("--delete_cache", help="delete cache", type=bool, default=True)
    parser.add_argument("--cache_path", help="cache path", default="CACHE")
    parser.add_argument("--mog_history", help="history", type=int, default=1000)
    parser.add_argument("--mog_var_threshold", help="var threshold", type=int, default=128)
    parser.add_argument("--mog_detect_shadows", help="detect shadows", type=bool, default=True)
    parser.add_argument("--show_video", help="show video", type=bool, default=True)
    parser.add_argument("--video_inference_scale", help="video inference scale", type=float, default=0.4)
    parser.add_argument("--start_timestamp", help="start frame", type=float, default=80.0)
    parser.add_argument("--end_timestamp", help="end frame", type=float, default=166.0)
    parser.add_argument("--confidence", type=float, default=0.7, help="Object confidence threshold")
    parser.add_argument("--cd_inference_width", type=int, default=1024, help="Change detection inference width")
    parser.add_argument("--skip_frames", type=int, default=0, help="Skip frames")

    args = parser.parse_args()

    if args.delete_cache:
        delete_folder(args.cache_path)

    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)


    if args.video_path.isdigit():
        print("Video path is a digit, converting to int.")
        args.video_path = int(args.video_path)


    # init model
    model, platform = setup_model(args)

    co_helper = COCO_test_helper(enable_letter_box=True)
    change_detector = change_detection.onnx_load_model("models/cd_1024.onnx")

    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    background_subtractor.setHistory(args.mog_history)
    background_subtractor.setVarThreshold(args.mog_var_threshold)
    background_subtractor.setDetectShadows(True)

    roi = ROI(args.roi_path)

    cap = cv2.VideoCapture(args.video_path)

    # Initialize YOLO model
    if not cap.isOpened():
        print(f"Unable to open: {args.video_path}")
        exit(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_saver = VideoSaver(fps, frame_size)

    frame_count = 0
    frame_count2 = 0
    start_frame_count = time.time()
    last_feature_timestamp = 0
    last_cut: float = 0
    filtered_boxes = []
    session_started = False
    close_last_video = False

    while True:
        success, img = cap.read()
        frame_count += 1

        if frame_count % 20 == 0:
            print(f"FPS: {frame_count / (time.time() - start_frame_count):.2f}")
            start_frame_count = time.time()
            frame_count = 0

        if success:
            frame_count += 1
            frame_count2 += 1
            current_timestamp = frame_count2 / fps
            if frame_count % 50 == 0:
                print(f"Frame count: {frame_count2}, FPS: {frame_count / (time.time() - start_frame_count):.2f}, Time: {current_timestamp:.02f}s")
                start_frame_count = time.time()
                frame_count = 0

            if current_timestamp < args.start_timestamp:
                continue
            if frame_count2 == args.start_timestamp * fps:
                print("Start frame reached")
                last_cut = current_timestamp
            if current_timestamp > args.end_timestamp:
                close_last_video=True
                break

            cv2.polylines(img, [np.array(roi.get_scaled_roi_polygon(img), np.int32)], True, (0, 255, 0), 2)

            cropped = roi.get_cropped_image2(img)
            cropped_and_translated = roi.get_cropped_and_translated_image3(img, args.cd_inference_width)
            # cropped2 = cv2.GaussianBlur(cropped2, (3, 3), 0)

            if not session_started:
                background_subtractor.apply(cropped_and_translated)

            video_saver.write(img)

            # skip frames to speed up the process
            if frame_count2 % 5 != 0:
                continue

            # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
            cropped = co_helper.letter_box(im=cropped.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
            #cropped_and_translated = co_helper.letter_box(im=cropped_and_translated.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            # cropped_and_translated = cv2.cvtColor(cropped_and_translated, cv2.COLOR_BGR2RGB)

            # preprocee if not rknn model
            if platform in ["pytorch", "onnx"]:
                input_data = img.transpose((2, 0, 1))
                input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
                input_data = input_data / 255.0
            else:
                input_data = np.expand_dims(cropped, 0)

            outputs = model.run([input_data])
            boxes, classes, scores = post_process(outputs)

            if boxes is not None:
                for box, score, cl in zip(boxes, scores, classes):
                    top, left, right, bottom = [int(_b) for _b in box]
                    score = float(score)
                    # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
                    cv2.rectangle(cropped, (top, left), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(cropped, "{0} {1:.2f}".format(CLASSES[cl], score), (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if cl == 0 and score > args.confidence:
                        if not session_started:
                            session_started = True
                            print("Session started")
                            current_background = background_subtractor.getBackgroundImage()

                        x1, y1, x2, y2 = top, left, right, bottom

                        document = {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": score,
                            "class_name": CLASSES[cl],
                            "timestamp": datetime.now().isoformat(),
                        }

                        last_feature_timestamp = current_timestamp
                        filtered_boxes.append(document)

            if session_started:
                print(current_timestamp, last_feature_timestamp, current_timestamp - last_feature_timestamp)
                if (current_timestamp - last_feature_timestamp) > 8:
                    current_background_rgb = cv2.cvtColor(current_background, cv2.COLOR_BGR2RGB)
                    cropped2_rgb = cv2.cvtColor(cropped_and_translated, cv2.COLOR_BGR2RGB)

                    start = time.time()
                    mask = change_detection.onnx_predict(change_detector, current_background_rgb, cropped2_rgb)
                    print(f"Outer inference time: {time.time() - start:.2f}s")
                    cv2.imwrite(os.path.join(args.cache_path, f"v{time.time():.2f}.png"), mask)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    cv2.imshow("mask", mask)
                    cv2.imshow("background", current_background)
                    cv2.imshow("new", cropped_and_translated)

                    background_subtractor = cv2.createBackgroundSubtractorMOG2()
                    background_subtractor.setHistory(args.mog_history)
                    background_subtractor.setVarThreshold(args.mog_var_threshold)
                    background_subtractor.setDetectShadows(True)
                    print("Session ended")
                    session_started = False
            else:
                if last_cut + normal_cut_time < current_timestamp:
                    write_file(filtered_boxes=filtered_boxes, start_time_text=video_saver.start_time_text, start_time=video_saver.start_time)
                    filtered_boxes.clear()
                    video_saver.cut_video()
                    last_cut = current_timestamp + 0

            if args.show_video:
                # img = cv2.resize(img, (0, 0), fx=args.video_inference_scale, fy=args.video_inference_scale)
                cv2.imshow("Video", img)
                cv2.imshow("Cropped", cropped)
                cv2.waitKey(1)

    if close_last_video:
        write_file(filtered_boxes=filtered_boxes, start_time_text=video_saver.start_time_text, start_time=video_saver.start_time)
        filtered_boxes.clear()

    video_saver.release()
    cap.release()
    cv2.destroyAllWindows()
