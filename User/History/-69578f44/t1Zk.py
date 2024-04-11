import time

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from LIDAR import Lidar


def scatter_plot_image(x, y):
    # set the maximum value for y axis to 2000
    x = np.array(x)
    y = np.array(y)

    fig, ax = plt.subplots()
    ax.set_ylim(0, 5000)
    ax.plot(x, y, 'o', markersize=2)

    # Convert plot to image
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)

    # Convert image to OpenCV format
    plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

    return plot


cmap = plt.get_cmap("rainbow")
angle_lower_limit_1 = 308
angle_upper_limit_1 = 383
angle_lower_limit_2 = 23
angle_upper_limit_2 = 37
first_selected = False
temp_angle = 0


def color_from_value(value, max_value=5000.0):
    value = min(max_value, max(0, value))  # make sure value is in range [0, 30000]
    norm_value = value / max_value  # normalize the value to the range [0, 1]
    color = cmap(norm_value)  # generate the color using the RdYlBu colormap
    # color = cm.RdYlBu(norm_value) # generate the color using the RdYlBu colormap
    return tuple(int(255 * x) for x in color[:3])  # convert the color to RGB and return as a tuple


def draw(distances, frame):
    # create a black image
    img = np.zeros((1000, 1000, 3), np.uint8)

    # draw a circle in the center of the image
    cv2.circle(img, (500, 500), 5, (255, 255, 255), -1)

    scale = 0.05

    # draw a line for each distance
    for angle, distance in distances:
        # convert distance and angle to x and y coordinates
        x = 500 + (distance * scale) * math.cos(math.radians(angle))
        y = 500 + (distance * scale) * math.sin(math.radians(angle))
        # draw a line from the center to the distance
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255) if angle_lower_limit_1 <= angle <= angle_upper_limit_1 or angle_lower_limit_2 <= angle <= angle_upper_limit_2 else (255, 0, 0), -1)

    # convert efficiently list of (x, y) to separate numpy arrays for x and y
    # order points by x
    distances = sorted([(x, y) for x, y in distances if 30 < y < 60000 and (angle_lower_limit_1 <= x <= angle_upper_limit_1 or angle_lower_limit_2 <= x <= angle_upper_limit_2)], key=lambda x: x[0])
    x, y = zip(*distances)

    y_interp = savgol_filter(y, 51, 3)
    n = 1280
    f = interp1d(x, y_interp, fill_value="extrapolate", kind="cubic")
    p = (angle_upper_limit_1 - angle_lower_limit_1) + (angle_upper_limit_2 - angle_lower_limit_2)
    # max_angle = max(distances, key=lambda x: x[0])[0]
    # min_angle = min(distances, key=lambda x: x[0])[0]
    x_interp = np.linspace(angle_lower_limit_1, p + angle_lower_limit_1, n)
    y_interp = f(x_interp)
    # plot = scatter_plot_image(x_interp, y_interp)

    # cv2.imshow("Lidar2", plot)

    first_row = [color_from_value(y) for _, y in zip(x_interp, y_interp)]
    image = np.array([first_row] * 380, dtype=np.uint8)
    cv2.imshow("Lidar3", image)

    # todo graphic bug, the arch is not drawn correctly when it goes from < 360 to > 0
    cv2.ellipse(img, (500, 500), (int(3000 * scale), int(3000 * scale)), 0, angle_lower_limit_1, angle_upper_limit_2, (255, 255, 255), 1)

    cv2.line(img, (500, 500), (int(500 + 3000 * scale * math.cos(math.radians(angle_lower_limit_1))), int(500 + 3000 * scale * math.sin(math.radians(angle_lower_limit_1)))), (255, 255, 255), 1)
    cv2.line(img, (500, 500), (int(500 + 3000 * scale * math.cos(math.radians(angle_upper_limit_2))), int(500 + 3000 * scale * math.sin(math.radians(angle_upper_limit_2)))), (255, 255, 255), 1)

    # display the image
    cv2.imshow("Lidar", img)
    cv2.imshow("Camera", frame)
    cv2.waitKey(1)


def print_angle(event, x, y, flags, param):
    global first_selected
    global angle_lower_limit_1
    global angle_upper_limit_1
    global angle_lower_limit_2
    global angle_upper_limit_2
    global temp_angle
    if event == cv2.EVENT_LBUTTONDOWN:
        angle = int(math.degrees(math.atan2(y - 500, x - 500)))
        if angle < 23.0:
            angle += 360
        if not first_selected:
            first_selected = True
            temp_angle = angle
        else:
            first_selected = False
            angle_lower_limit_1 = temp_angle
            angle_upper_limit_2 = angle

            mid = (angle - temp_angle) // 2
            angle_upper_limit_1 = temp_angle + mid + 1
            angle_lower_limit_2 = angle_upper_limit_1 - 1

            if temp_angle > angle:
                angle_upper_limit_1 = 383
                angle_lower_limit_2 = 23
            print(f"New limits: {angle_lower_limit_1}° - {angle_upper_limit_1}° and {angle_lower_limit_2}° - {angle_upper_limit_2}°")
        print(f"{angle}°")


if __name__ == "__main__":
    lidar = Lidar(port)
    cv2.namedWindow('Lidar')
    cv2.setMouseCallback("Lidar", lambda event, x, y, flags, param: print_angle(event, x, y, flags, param))
    cap = cv2.VideoCapture(1)
    lidar.active(True)
    # print image shape
    print("Image shape: ", cap.read()[1].shape)
    while True:
        if len(lidar.point_clouds) > 0:
            draw(lidar.get_distances(), frame=cap.read()[1])
            # print("Popped point cloud, remaining: ", len(lidar.point_clouds))
        time.sleep(0.02)
