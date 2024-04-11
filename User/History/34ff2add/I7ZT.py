# make discrete distribution from 0 to 10
# and plot the distribution

import numpy as np
import matplotlib.pyplot as plt

mean_1 = 3
std_1 = 2
mean_2 = 10
std_2 = 2

# set y limit to 0.2
# plt.ylim(0, 0.2)

a = 0.1

while True:

    plt.clf()
    # set limit
    plt.ylim(-14, 4)

    mean_1 += a
    if mean_1 > 15 or mean_1 < 0:
        a = -a

    # make discrete distribution
    x = np.arange(0, 16)
    y_1 = np.exp(-((x - mean_1) ** 2) / (2 * std_1**2)) / (std_1 * np.sqrt(2 * np.pi))
    y_2 = np.exp(-((x - mean_2) ** 2) / (2 * std_2**2)) / (std_2 * np.sqrt(2 * np.pi))

    # compute cross entropy between the two distributions
    cross_entropy = -np.sum(y_1 * np.log(y_2))
    print("Cross entropy between the two distributions: ", cross_entropy)

    # for every valule of x, compute surprise for distribution 1 and 2
    surprise_1 = np.log(y_1)
    surprise_2 = np.log(y_2)

    cross = y_1 * surprise_2

    # plot the two distributions together, the first in blue and the second in red as a bar chart
    plt.bar(x, surprise_2, color="blue")
    plt.bar(x, y_1 * 20, color="red", width=0.4)
    plt.bar(x, cross * 20, color="green", width=0.25)
    plt.show(block=False)
    plt.pause(0.1)