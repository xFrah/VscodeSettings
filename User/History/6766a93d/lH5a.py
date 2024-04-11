# generate random (x, y) points
# and plot them
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# generate 3 clusters of (x, y) points
# each cluster has 100 points
# each cluster has a different mean
# and a different standard deviation

# cluster 1
x1 = np.random.normal(0, 1, 100)
y1 = np.random.normal(0, 1, 100)

# cluster 2
x2 = np.random.normal(5, 1, 100)
y2 = np.random.normal(5, 1, 100)

# cluster 3
x3 = np.random.normal(10, 1, 100)
y3 = np.random.normal(10, 1, 100)

# get list of total points
x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))

# get 3 random points from the original list
# centroids = random.sample(list(zip(x, y)), 3)
# make centroids be [(4, 4), (5, 5), (6, 6)]
centroids = [(4, 4), (5, 5), (6, 6)]
# assign colors to centroids
colors = [
    (random.random(), random.random(), random.random()),
    (random.random(), random.random(), random.random()),
    (random.random(), random.random(), random.random()),
]

# make a dictionary of the random points as keys and empty lists as values


while True:
    assignments = {point: [] for point in centroids}

    # for each point, calculate the distance to the centroids and assign it to the closest centroid
    for point in zip(x, y):
        distances = []
        for centroid in assignments.keys():
            distances.append(math.sqrt((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2))
        assignments[centroids[distances.index(min(distances))]].append(point)

    # calculate the new centroids
    new_centroids = []
    for centroid in assignments.keys():
        new_centroids.append(
            (
                sum([point[0] for point in assignments[centroid]]) / len(assignments[centroid]),
                sum([point[1] for point in assignments[centroid]]) / len(assignments[centroid]),
            )
        )

    # check if the new centroids are the same as the old centroids
    # if they are, break the loop

    # plot the points in assignments with different random colors for each centroid
    for centroid, color in zip(assignments.keys(), colors):
        plt.scatter(
            [point[0] for point in assignments[centroid]],
            [point[1] for point in assignments[centroid]],
            color=color,
        )

    plt.scatter([point[0] for point in assignments.keys()], [point[1] for point in assignments.keys()], color="black")

    # plot arrow from old centroid to new centroid for each centroid
    for centroid, new_centroid in zip(assignments.keys(), new_centroids):
        plt.arrow(
            centroid[0],
            centroid[1],
            new_centroid[0] - centroid[0],
            new_centroid[1] - centroid[1],
            color="black",
            width=0.1,
            head_width=0.5,
        )

    # update centroids
    centroids = new_centroids

    if new_centroids == list(assignments.keys()):
        plt.show()
        break

    plt.show(block=False)
    plt.pause(0.1)
    time.sleep(1)
