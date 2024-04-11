import json
import os
from helpers import ROI
import natsort
import cv2


def init_mog(mog_history, mog_var_threshold):
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    background_subtractor.setHistory(mog_history)
    background_subtractor.setVarThreshold(mog_var_threshold)
    background_subtractor.setDetectShadows(True)
    return background_subtractor


def detect_changes(cache_path, roi_path, mog_history=1000, mog_var_threshold=128, delete_useless_videos=True):
    mog2 = init_mog(mog_history, mog_var_threshold)
    roi = ROI(roi_path)

    # get list of json files in path
    json_files = [f for f in os.listdir(cache_path) if f.endswith(".json")]
    json_files = natsort.natsorted(json_files)
    backgrounds = []
    session_list = []

    for json_file in enumerate(json_files):
        # read json file
        with open(os.path.join(cache_path, json_file), "r") as file:
            data = json.load(file)

        # check if data['sessions'] is empty
        session_list.append("sessions" in data and data["sessions"] != [])

        # open video
        video_file = json_file.replace(".json", ".mp4")
        cap = cv2.VideoCapture(os.path.join(cache_path, video_file))

        # try to open video
        if not cap.isOpened():
            print("Error opening video file", video_file)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        first_session = data["sessions"][0] if "sessions" in data and isinstance(data["sessions"], list) and data["sessions"] else None
        print("First session:", first_session)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video file", video_file, ", computing differences.")
                break

            frame_count += 1
            current_timestamp = frame_count / fps

            print(f"Frame {frame_count}, {current_timestamp} seconds.")

            if first_session:
                # check if we have reached the next session
                if current_timestamp >= first_session["start"]:
                    print("First session start:", first_session["start"], ", getting background image.")
                    backgrounds.append(mog2.getBackgroundImage())

            frame = roi.get_cropped_and_translated_image3(frame)
            mog2.apply(frame)

    for i in range(len(json_files)):
        if session_list[i] == False and session_list[i + 1] == False and (i != 0 and session_list[i - 1] == False):
            video_file = json_files[i].replace(".json", ".mp4")
            os.remove(os.path.join(cache_path, json_files[i]))
            print("Removed", json_files[i])
            os.remove(os.path.join(cache_path, video_file))
            print("Removed", video_file)
