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
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_text = os.path.join(args.cache_path, f"v{self.start_time:.2f}.mp4")
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


def main(video_path, roi_path) -> None:
    change_detector = change_detection.load_model("models/vgg16bn_iade_4_deeplabv3_PCD.pth")

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
        # img_t = img.copy()

        if success:
            frame_count += 1
            frame_count2 += 1
            current_timestamp = frame_count2 / fps
            if frame_count % 50 == 0:
                print(f"Frame count: {frame_count2}, FPS: {frame_count / (time.time() - start_frame_count):.2f}, Time: {current_timestamp:.02f}s")
                start_frame_count = time.time()
                frame_count = 0

            if current_timestamp < start_timestamp:
                continue
            if frame_count2 == start_timestamp * fps:
                print("Start frame reached")
                last_cut = current_timestamp
            if current_timestamp > end_timestamp:
                break

            # cv2.polylines(img, [np.array(roi.get_scaled_roi_polygon(img), np.int32)], True, (0, 255, 0), 2)

            cropped = roi.get_cropped_image2(img)
            cropped_and_translated = roi.get_cropped_and_translated_image3(img, cd_inference_width)
            # cropped2 = cv2.GaussianBlur(cropped2, (3, 3), 0)

            if not session_started:
                background_subtractor.apply(cropped_and_translated)

            video_saver.write(img)

            # skip frames to speed up the process
            if frame_count2 % 3 != 0:
                continue

            results = model(cropped, imgsz=480, stream=True, verbose=False)

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
                if (current_timestamp - last_feature_timestamp) > 8:
                    current_background_rgb = cv2.cvtColor(current_background, cv2.COLOR_BGR2RGB)
                    cropped2_rgb = cv2.cvtColor(cropped_and_translated, cv2.COLOR_BGR2RGB)

                    start = time.time()
                    mask = change_detection.predict(change_detector, current_background_rgb, cropped2_rgb)
                    print(f"Outer inference time: {time.time() - start:.2f}s")
                    cv2.imwrite(os.path.join(args.cache_path, f"v{time.time():.2f}.png"), mask)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    final = np.vstack([current_background, cropped_and_translated, mask])
                    cv2.imshow("Final", final)

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
                img = cv2.resize(img, (0, 0), fx=video_inference_scale, fy=video_inference_scale)
                cv2.imshow("Video", img)
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
z

    main(args.video_path, args.roi_path)
