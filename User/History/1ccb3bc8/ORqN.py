import time
import matplotlib.pyplot as plt
import numpy as np


# generate point with covariance 0.8
def generate_point():
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)
    return [x, y]


# create 2 subplots with limits -3.5 to 3.5 on both axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-3.5, 3.5)

cov = np.array([[1, 0], [0, 1]])


# generate 100 points
points = [generate_point() for i in range(100)]


while True:

    plt.clf()
    # set titles
    ax2.set_title(f"Applying the covariance matrix (cov = {cov[1][0]:.0f})")

    # apply the covariance matrix to the points
    points = [cov @ p for p in points]

    # scatter the points in red on ax1
    ax1.scatter([p[0] for p in points], [p[1] for p in points], color="red")

    # plot unit vectors
    plt.quiver(0, 0, cov[0][0], cov[1][0], angles="xy", scale_units="xy", scale=1)
    plt.quiver(0, 0, cov[0][1], cov[1][1], angles="xy", scale_units="xy", scale=1)

    plt.show(block=False)
    plt.pause(0.1)

    cov[1][0] += 0.05
    cov[0][1] += 0.05
    
    if cov[1][0] > 0.7:
        time.sleep(10)

    # plt.savefig("images/" + str(time.time()) + ".png")
