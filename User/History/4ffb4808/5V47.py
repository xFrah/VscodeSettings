import numpy as np
from ultralytics import YOLO
import cv2
import math
from datetime import datetime
import time
import change_detection
import json
import os
from helpers import classNames, ROI, delete_folder
from skimage import exposure


normal_cut_time: int = 10
working: bool = True
clean_video: bool
show_video: bool
video_inference_scale: float
start_frame: int
end_frame: int


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
        print("Start time:", self.start_time_text)
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")
        print("Frame size:", self.frame_size)
        self.video_text = os.path.join(args.cache_path, f"v{self.start_time:.2f}.mp4")
        print("Video text:", self.video_text)
        self.video_output = cv2.VideoWriter(self.video_text, self.codec, self.fps, self.frame_size)

    def __del__(self):
        try:
            self.video_output.release()
        except Exception as e:
            pass


def write_file(filtered_boxes, start_time_text, start_time) -> None:
    empty = len(filtered_boxes) == 0
    features_elem = {
        "empty": empty,
        "ts_init": start_time_text,
        "ts_final": datetime.now().isoformat(),
        "features": filtered_boxes,
    }

    json_text = os.path.join(args.cache_path, f"v{start_time:.2f}.json")
    if not os.path.exists(json_text):
        with open(json_text, "w") as json_file:
            json.dump(features_elem, json_file)

    print("File written")


def adjust_v_channel(img1, img2):
    # Convert images from BGR to HSV
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Compute the mean of the V channel for each image
    v_mean1 = np.mean(hsv1[:, :, 2])
    v_mean2 = np.mean(hsv2[:, :, 2])

    # Calculate the difference
    v_diff = v_mean2 - v_mean1

    # Adjust the V channel of img1
    adjusted_v1 = np.clip(hsv1[:, :, 2].astype(np.float32) + v_diff, 0, 255).astype(np.uint8)

    # Update the V channel in the first HSV image
    adjusted_hsv1 = hsv1.copy()
    adjusted_hsv1[:, :, 2] = adjusted_v1

    # Convert back to BGR for display or saving
    adjusted_img1 = cv2.cvtColor(adjusted_hsv1, cv2.COLOR_HSV2BGR)

    return adjusted_img1


def main(video_path, roi_path) -> None:
    change_detector = change_detection.load_model("vgg16bn_iade_4_deeplabv3_PCD.pth")

    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    background_subtractor.setHistory(args.mog_history)
    background_subtractor.setVarThreshold(args.mog_var_threshold)
    background_subtractor.setDetectShadows(True)

    roi = ROI(roi_path)

    cap = cv2.VideoCapture(video_path)

    # Initialize YOLO model
    if not cap.isOpened():
        print(f"Unable to open: {video_path}")
        exit(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_saver = VideoSaver(fps, frame_size)

    model = YOLO("yolo-Weights/yolov8n.pt")
    frame_count = 0
    frame_count2 = 0
    start_frame_count = time.time()
    last_feature_timestamp = 0
    last_cut: float = 0
    filtered_boxes = []
    session_started = False
    while True:
        success, img = cap.read()
        img_t = img.copy()

        if success:
            frame_count += 1
            frame_count2 += 1
            current_timestamp = frame_count2 / fps
            if frame_count % 50 == 0:
                print(f"Frame count: {frame_count2}, FPS: {frame_count / (time.time() - start_frame_count):.2f}, Time: {current_timestamp:.02f}s")
                start_frame_count = time.time()
                frame_count = 0

            cropped = roi.get_cropped_image2(img)

            if not session_started:
                background_subtractor.apply(cropped)

            if current_timestamp < start_timestamp:
                continue
            if frame_count2 == start_timestamp * fps:
                print("Start frame reached")
                last_cut = current_timestamp
            if current_timestamp > end_timestamp:
                break

            video_saver.write(img)

            # skip frames to speed up the process
            if frame_count2 % 3 != 0:
                continue

            results = model(cropped, imgsz=480, stream=True, verbose=False)

            cv2.polylines(img_t, [np.array(roi.get_scaled_roi_polygon(img), np.int32)], True, (0, 255, 0), 2)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    if cls == 0 and conf > confidence:
                        if not session_started:
                            session_started = True
                            print("Session started")
                            current_background = background_subtractor.getBackgroundImage()
                        cls = int(box.cls[0])

                        document = {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": math.ceil((box.conf[0] * 100)) / 100,
                            "class_name": classNames[cls],
                            "timestamp": datetime.now().isoformat(),
                        }

                        last_feature_timestamp = current_timestamp
                        filtered_boxes.append(document)

            if session_started:
                if (current_timestamp - last_feature_timestamp) > 3:
                    current_background = cv2.cvtColor(current_background, cv2.COLOR_BGR2RGB)
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

                    current_background = cv2.resize(current_background, (current_background.shape[1] * 2, current_background.shape[0]))
                    cropped = cv2.resize(cropped, (cropped.shape[1] * 2, cropped.shape[0]))
                    # show current background and cropped image
                    cv2.imshow("Background", current_background)
                    cv2.imshow("Cropped2", cropped)
                    mask = change_detection.predict(change_detector, current_background, cropped)
                    cv2.imwrite(os.path.join(args.cache_path, f"v{time.time():.2f}.png"), mask)
                    cv2.imshow("Mask", mask)

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

            if show_video:
                img_t = cv2.resize(img_t, (0, 0), fx=video_inference_scale, fy=video_inference_scale)
                cv2.imshow("Video", img_t)
                cv2.imshow("Cropped", cropped)

            if cv2.waitKey(1) == ord("q"):
                break
        else:
            print("Video file not found or is empty.")
            break

    video_saver.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Change detection")
    parser.add_argument("--video_path", help="path to video file", default="Sedili.mp4")
    parser.add_argument("--roi_path", help="path to roi file", default="ROI/1/ROI.csv")
    parser.add_argument("--delete_cache", help="delete cache", type=bool, default=True)
    parser.add_argument("--cache_path", help="cache path", default="CACHE")
    parser.add_argument("--mog_history", help="history", type=int, default=1000)
    parser.add_argument("--mog_var_threshold", help="var threshold", type=int, default=128)
    parser.add_argument("--mog_detect_shadows", help="detect shadows", type=bool, default=True)
    parser.add_argument("--show_video", help="show video", type=bool, default=True)
    parser.add_argument("--video_inference_scale", help="video inference scale", type=float, default=0.4)
    parser.add_argument("--start_timestamp", help="start frame", type=float, default=60.0)
    parser.add_argument("--end_timestamp", help="end frame", type=float, default=166.0)
    parser.add_argument("--conf", type=float, default=0.7, help="Object confidence threshold")
    args = parser.parse_args()

    print("Video path:", args.video_path)
    print("ROI path:", args.roi_path)
    print("Delete cache:", args.delete_cache)
    print("Cache path:", args.cache_path)
    print("MOG history:", args.mog_history)
    print("MOG var threshold:", args.mog_var_threshold)
    print("MOG detect shadows:", args.mog_detect_shadows)
    print("Show video:", args.show_video)
    print("Video inference scale:", args.video_inference_scale)
    print("Start frame:", args.start_timestamp)
    print("End frame:", args.end_timestamp)
    print("Confidence:", args.conf)

    if args.delete_cache:
        delete_folder(args.cache_path)

    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)

    show_video = args.show_video
    video_inference_scale = args.video_inference_scale
    start_timestamp = args.start_timestamp
    end_timestamp = args.end_timestamp
    confidence = args.conf

    if args.video_path.isdigit():
        print("Video path is a digit, converting to int.")
        args.video_path = int(args.video_path)

    main(args.video_path, args.roi_path)
