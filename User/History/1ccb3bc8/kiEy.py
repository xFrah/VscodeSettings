import os
import time
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio


# generate point with covariance 0.8
def generate_point():
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)
    return [x, y]


# set limits 3.5
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)

cov = np.array([[1.0, 0.7], [0.7, 1.0]])


# generate 100 points
o_points = [generate_point() for i in range(100)]

# plt.savefig("images/" + str(time.time()) + ".png")

# make gif
images = []
for filename in os.listdir("images"):
    images.append(iio.imread("images/" + filename))
#   pass
# make gif from images
iio.imwrite("cov.gif", images, duration=75, loop=0)
print("gif saved")

while True:
    pass


while True:
    plt.clf()
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    # set titles
    plt.title(f"Applying the inverse covariance matrix (cov = {cov[1][0]:.1f})")

    # apply the covariance matrix to the points
    points = [cov @ p for p in o_points]

    # scatter the points in red on ax1
    plt.scatter([p[0] for p in points], [p[1] for p in points], color="red")

    # plot unit vectors
    plt.quiver(0, 0, cov[0][0], cov[1][0], angles="xy", scale_units="xy", scale=1)
    plt.quiver(0, 0, cov[0][1], cov[1][1], angles="xy", scale_units="xy", scale=1)

    plt.show(block=False)
    plt.pause(0.05)

    cov[1][0] = cov[1][0] - 0.01
    cov[0][1] = cov[0][1] - 0.01

    print(cov[1][0])

    plt.savefig("images/" + str(time.time()) + ".png")

    if cov[1][0] < 0:
        break
