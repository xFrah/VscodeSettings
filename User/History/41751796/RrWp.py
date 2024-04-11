import math
import os
import numpy as np
import matplotlib.pyplot as plt

# plot probability distribution
n = 4
s = 3
mean = 0.25
i = 1

while False:
    plt.clf()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    n += s
    std = math.sqrt((mean * (1 - mean)) / n)
    if n > 400:
        s *= -1
    elif n < 4:
        s *= -1
        break
    x = np.linspace(0, 1, 100)
    # get probability distribution of sample proportion with sample size n and sample mean 0.25
    y = np.exp(-((x - mean) ** 2) / (2 * std**2)) / (std * math.sqrt(2 * math.pi))
    # the sum of all probabilities should be 1
    y = y / np.sum(y)
    print("n = ", n, "mean = ", mean, "std = ", std)
    plt.plot(x, y, label="normal distribution")
    # add legend to plot, it must have n, mean and std
    plt.legend(loc="upper left", title=f"n = {n}, mean = {mean}, std = {std:.3f}")
    plt.show(block=False)
    plt.pause(0.05)
    # save plot as image
    plt.savefig(f"images/{i}.png")
    i += 1

# create gif from images
import imageio.v3 as iio

images = []
for filename in sorted(os.listdir("images")):
    images.append(iio.imread("images/" + filename))

# reorder images because they are in incorrect order


# make gif from images
iio.imwrite("kmeans3.gif", images, duration=50, loop=0)
