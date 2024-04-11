# import subprocess
import json
import threading
import drone_helpers
import time
from land import FootballField

import tornado.ioloop
import tornado.web
import tornado.websocket
import requests
import time

drone = None
tornado_opened = False
markers = None
i = 0


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
        }
        if markers:
            msg["markers"] = markers
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

drone = drone_helpers.Drone("tcp:127.0.0.1:5762", start=False)
coords = [
    (12.455002, 41.934481),
    (12.454218, 41.934329),
    (12.454537, 41.933425),
    (12.455324, 41.933574),
]
# coords = [(y, x) for x, y in coords]

field = FootballField(coords)
print(field)
markers = []

while True:
    # when i press enter in command line
    msg = input("Press Enter to continue...")
    if msg == "exit" or msg == "quit" or msg == "q" or msg == "stop":
        break
    elif msg == "measure":
        # send http get request to the server
        res = requests.get("http://127.0.0.1:3000/measure")
        markers.append(drone.location[:2])


# waypoints_list[0](master)

master.close()
