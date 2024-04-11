# import subprocess
import time
from drone_helpers import Drone, progress_debug

# from helpers import check_if_window_exists_with_title_starting_with

from land import FootballField
import webbrowser
import folium
import numpy as np
import cv2
from helpers import find_shortest_path
from waypoints import LandWaypoint

coords = [
    (41.943033, 12.541372),
    (41.943017, 12.541581),
    (41.942820, 12.541336),
    (41.942822, 12.541599),
]
coords = [(y, x) for x, y in coords]

field = FootballField(coords)
print(field)

h = 500

# open football field color cv2 image and plot it, with callback. it must have h height and 1.67h width.
image = np.zeros((h, int(h * 1.67), 3), np.uint8)
image[:] = (0, 140, 20)
markers = []


# mouse callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        x_frac = x / image.shape[1]
        y_frac = 1 - (y / image.shape[0])
        markers.append((x_frac, y_frac))
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
markers = [field.frac_to_coord(*marker)[::-1] for marker in markers]
permutation, distance = find_shortest_path(markers)
print(permutation)
print(f"Distance traveled: {distance * 111000:.2f}m")
# reorder markers
markers = [markers[i] for i in permutation]

for i, marker in enumerate(markers):
    folium.Marker(tuple(marker), popup=f"Waypoint {i}").add_to(fmap)
    if i != 0:
        folium.PolyLine(
            [markers[i - 1], marker], color="red", weight=2.5, opacity=1
        ).add_to(fmap)

outfp = r"map.html"

fmap.save(outfp)

webbrowser.open(outfp)

# --home=41.934275,12.633201,17,353
# if not check_if_window_exists_with_title_starting_with("Copter"):
#     command = r"title Copter && dronekit-sitl copter --home=41.934654,12.454560,17,353"
#     proc = subprocess.Popen(f'start cmd /k "{command}"', shell=True)
# if not check_if_window_exists_with_title_starting_with("Mavproxy"):
#     command = r"title Mavproxy && mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:1450 --out udp:127.0.0.1:1440"
#     proc = subprocess.Popen(f'start cmd /k "{command}"', shell=True)

# mavproxy.py --master "COM6" --baudrate=115200 --out udp:127.0.0.1:1450 --out udp:127.0.0.1:1440 --out: udp:192.168.137.255:1750

drone = Drone("udp:127.0.0.1:1450")
waypoints_list = [LandWaypoint(*marker, debug=True) for marker in markers]

for waypoint in waypoints_list:
    print(f"Running {waypoint}")
    status = "TRAVELING"
    waypoint(drone)  # run waypoint

    while True:
        finished, failed = waypoint.has_finished()
        if finished:
            print(f"{waypoint} {'failed' if failed else 'succeeded'}")
            if failed:
                quit(1)
            break
        progress_debug(drone, waypoint)
        time.sleep(waypoint.finished_condition_polling)
    time.sleep(2)

# waypoints_list[0](master)

drone.close()

# mavproxy.py --master "COM6" --baudrate=115200 --out udp:127.0.0.1:1450 --out udp:127.0.0.1:1440 --out udp:192.168.137.255:1750