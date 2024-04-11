import glob
from ultralytics import YOLO
import cv2
import math
import csv
from pymongo import MongoClient
from datetime import datetime
import time
import threading
from helpers import point_inside_polygon


def send_to_mongo():
    global filtered_boxes
    while True:
        if filtered_boxes:
            features_elem = {"timestamp": datetime.now(), "features": filtered_boxes}

            # Inserimento del documento nella collezione MongoDB
            collection.insert_one(features_elem)

            # Pulizia della lista dopo l'inserimento
            filtered_boxes = []

        time.sleep(1)


# Select video and ROI
roi_id = 1
video_id = 11

# init
filtered_boxes = []
features_elem = []

# thread
mongo_thread = threading.Thread(target=send_to_mongo)
mongo_thread.daemon = True
mongo_thread.start()

# MongoDB
client = MongoClient("localhost", 27017)
db = client["surveillance"]
collection = db["features"]

# videos=['CASSONETTI/1/A240214_130834_130848.265.mp4', 'CASSONETTI/1/A240214_145315_145329.265.mp4', 'CASSONETTI/1/A240214_145333_145347.265.mp4', 'CASSONETTI/1/A240214_145352_145406.265.mp4']
file_list = glob.glob("CASSONETTI/*/*")
file_list = sorted(file_list)

roi_list = glob.glob("ROI/*/*")


# ROI
roi_polygon = []
with open(roi_list[roi_id], newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        roi_polygon.append((int(row[0]), int(row[1])))

# start webcam
# cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture(file_list[video_id])
cap.set(3, 640)
cap.set(4, 480)

# TODO try to reduce the parameters of YOLO, by removing classes, pruning and quantization

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
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
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

while True:
    success, img = cap.read()

    if success:
        img = cv2.resize(img, (1280, 720))

        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2

                cls = int(box.cls[0])

                if point_inside_polygon(x_center, y_center, roi_polygon) and (cls == 0 or cls == 2):
                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    confidence_str = "{:.2f}".format(confidence)
                    box_text = "{} {}".format(classNames[cls], confidence_str)

                    cv2.putText(img, box_text, org, font, fontScale, color, thickness)

                    document = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": math.ceil((box.conf[0] * 100)) / 100,
                        "class_name": classNames[cls],
                        "timestamp": datetime.now(),
                    }
                    filtered_boxes.append(document)

            # if filtered_boxes:
            #     features_elem = {
            #             'timestamp': datetime.now(),
            #             'features': filtered_boxes,
            #     }
            #     collection.insert_one(features_elem)
            #     filtered_boxes = []
            #     features_elem =[]

        # cv2.imshow('Webcam', img)
        cv2.imshow("Video", img)
        if cv2.waitKey(1) == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
