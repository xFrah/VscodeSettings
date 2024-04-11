import threading
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time
import datetime
import paho.mqtt.client as mqtt
import config

padiglione_dict = config.get_padiglione_dict()
counter = 0  # Counter to keep track of the number of people in the room


def run_tracker_in_thread(source, padiglione, invert_flag):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        source (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        padiglione (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
    Press 'q' to quit the video display window.
    """
    print(f"[INFO] Thread {threading.current_thread().name} is alive.")
    model = YOLO("best.pt")
    # Store the track history
    det_dict = {}

    frame_counter = 0  # Counter to keep track of the frame number
    if source.isdigit():
        source = int(source)
    video = cv2.VideoCapture(source)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames
        frame_counter += 1  # Increment the frame counter

        if not ret:
            time.sleep(0.3)  # TODO add timeout
            continue

        # Track objects in frames if available
        results = model.track(frame, persist=True, max_det=300, classes=0, conf=0.3)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except AttributeError:
            track_ids = []

        # Update the last seen frame number for each ID
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            if track_id not in det_dict:
                det_dict[track_id] = [
                    datetime.datetime.now(),  # last seen
                    (x, y),  # first appeared position
                    (x, y),  # last appeared position
                ]
            else:
                det_dict[track_id][0] = datetime.datetime.now()
                det_dict[track_id][2] = (x, y)

            x = int(x - (w / 2))
            y = int(y - (h / 2))
            frame = cv2.rectangle(frame, (x, y), (int(x + w), int(y + h)), (0, 0, 255), 2)

        delta = datetime.timedelta(seconds=3)
        line_x = int(0.5 * frame.shape[1])
        frame = cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 3)

        for person, (first_appearance, init_position, last_position) in list(det_dict.items()):
            side0 = "right" if init_position[0] > line_x else "left"
            side1 = "right" if last_position[0] > line_x else "left"
            if side0 != side1:
                # print(f"{person}-AAAA {last_position}")
                cv2.circle(frame, (int(last_position[0]), int(last_position[1])), 12, (0, 255, 0), 2)
            if datetime.datetime.now() - first_appearance > delta:  # has it disappeared?
                if side0 != side1:
                    padiglione_dict[padiglione] += 1 if side1 == "right" else -1
                    # print(f"{person} joined." if side1 == "right" else f"{person} left")
                del det_dict[person]

        # Draw the state of the invert_flag on the window
        flag_text = "Inverted" if invert_flag else "Normal"
        cv2.putText(frame, flag_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Draw the counter on the window
        cv2.putText(frame, f"Counter: {padiglione_dict[padiglione]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(f"Tracking_Stream_{padiglione}", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release video sources
    video.release()


def main():
    conf = config.get_config()

    tracker_threads = [
        threading.Thread(target=run_tracker_in_thread, args=(conf["source"], iid, conf["inverted"]), daemon=True, name=f"Tracker_{iid}")
        for iid, conf in config.get_padiglioni_config().items()
    ]

    for thread in tracker_threads:
        thread.start()

    try:
        client = mqtt.Client(client_id="mqtt_user", clean_session=True, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
        client.username_pw_set("mqtt_user", "Beam2020")
        client.connect("homeassistant.local", port=1883, keepalive=100)
    except Exception as e:
        print(f"[ERROR] MQTT connection failed. {e}")
        quit()

    while True:
        time.sleep(0.1)
        for key in padiglione_dict:
            buf = padiglione_dict[key]
            client.publish(f"pad{key}", buf)
            print(f"[LOG] Sent {key}:{buf}")

        

    while True:
        # save the shit
        for key in padiglione_dict:
            buf = padiglione_dict[key]
            with open("padiglioni/" + key + ".txt", "w") as f:
                f.write(str(buf))
            client.publish(f"pad{key}", buf)
            print(f"[LOG] Sent {key}:{buf}")
            print(f"[INFO] Saved {key}:{buf}")
        mqtt_client.update_handler()
            time.sleep(0.1)

    for thread in tracker_threads:
        thread.join()

    print("[INFO] Finished processing all streams.")
    # Clean up and close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
