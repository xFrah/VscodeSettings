import math
import numpy as np
import matplotlib.pyplot as plt

# plot probability distribution
n = 4
s = 1
mean = 0.25

while True:
    plt.clf()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    n += s
    std = math.sqrt((mean * (1 - mean)) / n)
    if n > 400:
        s = -1
    elif n < 4:
        s = 1
    x = np.linspace(0, 1, 100)
    # get probability distribution of sample proportion with sample size n and sample mean 0.25
    y = np.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))
    # the sum of all probabilities should be 1
    y = y / np.sum(y)
    # now we generate random samples from the probability distribution and we should get a normal distribution
    # with mean 0.25 and standard deviation sqrt((mean * (1 - mean)) / n)) without knowing the population distribution
    # the histogram of the data should amount to 1
    print("n = ", n, "mean = ", mean, "std = ", std)
    plt.plot(x, y, label="normal distribution")
    plt.show(block=False)
    plt.pause(0.05)
