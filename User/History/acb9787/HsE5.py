# make 2d plane in 3d space and plot it

import time
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

ax.set_zlim3d(0, 1)

# add point cloud

x = np.random.rand(100)

y = np.random.rand(100)

z = np.zeros(100)

# plot the point cloud

ax.scatter(x, y, z, c="b", marker="o")


def transform(matrix, vector):
    return np.dot(matrix, vector)


# transform the point cloud

matrix = np.array(
    [
        [1, 0, ],
        [0, 1, 0],
        [1, 0, 1],
    ]
)

x1, y1, z1 = transform(matrix, np.array([x, y, z]))

# label axes

ax.set_xlabel("X")

ax.set_ylabel("Y")

ax.set_zlabel("Z")

ax.scatter(x1, y1, z1, c="r", marker="o")


# plot arrow that goes from origin to point
def plot_arrow(ax, x, y, z, origin):
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        x,
        y,
        z,
        length=1,
        arrow_length_ratio=0.1,
        pivot="tail",
        color="k",
    )


# get columns in matrix
for row in matrix.T[:]:
    plot_arrow(ax, row[0], row[1], row[2], [0, 0, 0])

plt.show()

time.sleep(10)
