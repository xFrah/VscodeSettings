import json
import time
from scipy.spatial import distance as dist
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
import pygetwindow as gw
import paho.mqtt.client as mqtt


def close_windows_with_title_starting_with(prefix):
    all_windows = gw.getAllWindows()
    for window in all_windows:
        if window.title.startswith(prefix):
            window.close()


def check_if_window_exists_with_title_starting_with(prefix):
    all_windows = gw.getAllWindows()
    for window in all_windows:
        if window.title.startswith(prefix):
            return True
    return False


def order_points(pts):
    pts = np.array(pts)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return [(x, y) for x, y in [tl, tr, br, bl]]


def sort_coordinates(list_of_xy_coords):
    list_of_xy_coords = np.array(list_of_xy_coords)
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(-angles)
    return list_of_xy_coords[indices]


def find_rectangle_bases_points(pts):
    """
    Given a rectangle, find the order of the points that make the point 0 the left point of one of the two bases of the rectangle.
    """
    valid_perms = []
    perms = cyclic_permutations(pts)
    # if distance from the first point and the second point is greater than distance from second point and third point, keep the permutation
    for perm in perms:
        if dist.euclidean(perm[0], perm[1]) > dist.euclidean(perm[1], perm[2]):
            valid_perms.append(perm)
    return valid_perms


def cyclic_permutations(array):
    return [np.roll(array, -i, axis=0) for i in range(len(array))]


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    if rc == 0:
        client.subscribe("measures")


def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_disconnect(client, userdata, rc):
    print("Disconnected with result code " + str(rc))


def connect_to_mqtt_and_give_measures(host: str, port=1883, how_much=5):
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        buffer = []

        def on_message(client, userdata, msg):
            print(msg.topic + " " + str(msg.payload))
            try:
                buffer.append(json.loads(msg.payload.decode()))
            except Exception as e:
                print(e)

        client.on_message = on_message
        client.on_subscribe = on_subscribe
        client.on_disconnect = on_disconnect
        client.connect(host, port, 60)
        client.subscribe("measures")
        client.loop_start()
        client.publish("commands", "command:pullup")
        time.sleep(2)
        start = time.time()
        while time.time() - start < how_much:
            time.sleep(0.5)
        client.disconnect()
        client.loop_stop()
        return buffer[-1]
        # new = [int(x.decode()) for x in buffer]
        # mean = sum(new) / len(new)
        # return mean
    except Exception as e:
        return print(e)


def find_shortest_path(waypoints):
    """
    Given a list of waypoints, find the shortest path that visits all of them.
    """
    if len(waypoints) < 3:
        return list(range(len(waypoints))), 0
    # create a distance matrix
    distance_matrix = dist.cdist(waypoints, waypoints, "euclidean")
    # solve the tsp
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    # return the permutation
    return permutation, distance
