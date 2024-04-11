# import subprocess
import json
from random import random
import threading
import drone_helpers

# from helpers import check_if_window_exists_with_title_starting_with

import time
from land import FootballField
import webbrowser
import folium
import numpy as np
import cv2
from helpers import find_shortest_path, connect_to_mqtt_and_give_measures
from waypoints import LandWaypoint

import tornado.ioloop
import tornado.web
import tornado.websocket
import time

drone = None
tornado_opened = False
markers = None
json_markers = None
status = "TRAVELING"
i = 0


class JsonWaypoint:
    def __init__(self, lat, lon, id):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.humidity: float = None
        self.temperature: float = None

    def to_json(self):
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "humidity": self.humidity,
            "temperature": self.temperature,
        }


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        # Allow all cross-origin traffic
        return True

    def open(self):
        print("open success")
        # timer that sends data to the front end once per second
        self.timer = tornado.ioloop.PeriodicCallback(self.send_data, 333)
        self.timer.start()

    def on_close(self):
        self.timer.stop()

    def send_data(self):
        if drone is None:
            return
        # send the current time to the front end
        lat, lon, alt = drone.location
        msg = {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "current_waypoint": i,
            "status": status,
        }
        if markers:
            msg["markers"] = [
                JsonWaypoint(*e, id=i).to_json() for i, e in enumerate(markers)
            ]
        self.write_message(json.dumps(msg).encode("utf-8"))


# --home=41.934275,12.633201,17,353
# if not check_if_window_exists_with_title_starting_with("Copter"):
#     command = r"title Copter && dronekit-sitl copter --home=41.934654,12.454560,17,353"
#     proc = subprocess.Popen(f'start cmd /k "{command}"', shell=True)
# if not check_if_window_exists_with_title_starting_with("Mavproxy"):
#     command = r"title Mavproxy && mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:1450 --out udp:127.0.0.1:1440"
#     proc = subprocess.Popen(f'start cmd /k "{command}"', shell=True)


def mini_websocket_thread():
    application = tornado.web.Application([(r"/ws", WebSocketHandler)])

    application.listen(3001)
    print("-------------mini websocket thread started")
    global tornado_opened
    tornado_opened = True
    tornado.ioloop.IOLoop.instance().start()


threading.Thread(target=mini_websocket_thread, daemon=True).start()

# wait for websocket to open
while not tornado_opened:
    time.sleep(0.5)

drone = drone_helpers.Drone("tcp:127.0.0.1:5762")
while True:
    coords = [
        (12.455002, 41.934481),
        (12.454218, 41.934329),
        (12.454537, 41.933425),
        (12.455324, 41.933574),
    ]
    # coords = [(y, x) for x, y in coords]

    field = FootballField(coords)
    print(field)

    h = 500

    # open football field color cv2 image and plot it, with callback. it must have h height and 1.67h width.
    image = np.zeros((h, int(h * 1.67), 3), np.uint8)
    image[:] = (0, 140, 20)
    markers_ = []

    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            x_frac = x / image.shape[1]
            y_frac = 1 - (y / image.shape[0])
            markers_.append((x_frac, y_frac))
            cv2.circle(image, (x, y), 30, (255, 0, 0), -1)

    # create cv2 window with click callback
    cv2.namedWindow("Football field")
    cv2.setMouseCallback("Football field", draw_circle)

    edges = []

    while True:
        for edge in edges:
            cv2.line(image, edge[0], edge[1], (0, 0, 0), 3)
        cv2.imshow("Football field", image)
        k = cv2.waitKey(1) & 0xFF
        # if enter is detected
        if k == 13:
            # close window
            cv2.destroyAllWindows()
            break

    print(field.gdf.crs)

    # plot contour from vertices
    fmap = field.gdf.explore()
    markers_ = [field.frac_to_coord(*marker)[::-1] for marker in markers_]
    permutation, distance = find_shortest_path(markers_)
    print(permutation)
    print(f"Distance traveled: {distance * 111000:.2f}m")
    # reorder markers
    markers = [markers_[i] for i in permutation]
    json_markers = [JsonWaypoint(*e, id=i).to_json() for i, e in enumerate(markers)]

    for i, marker in enumerate(markers):
        folium.Marker(tuple(marker), popup=f"Waypoint {i}").add_to(fmap)
        if i != 0:
            folium.PolyLine(
                [markers[i - 1], marker], color="red", weight=2.5, opacity=1
            ).add_to(fmap)

    outfp = r"map.html"

    fmap.save(outfp)

    # webbrowser.open(outfp)

    waypoints_list = [LandWaypoint(*marker, debug=True) for marker in markers]  # type: ignore

    i = 0
    for waypoint in waypoints_list:
        print(f"Running {waypoint}")
        status = "TRAVELING"
        waypoint(drone)  # run waypoint

        while True:
            finished, failed = waypoint.has_finished()
            if finished:
                print(f"{waypoint} {'failed' if failed else 'succeeded'}")
                break
            drone_helpers.progress_debug(drone, waypoint)
            time.sleep(waypoint.finished_condition_polling)
        print(connect_to_mqtt_and_give_measures("5.196.23.212"))
        json_markers[i].humidity = 40 + (random() * 5)
        status = "MEASURING"
        time.sleep(5)
        i += 1

# waypoints_list[0](master)

master.close()
