from matplotlib import pyplot as plt
import numpy as np

means = [1.80, 1.13, 1.05, 1.12]
stds = [46.81, 6.91, 3.31, 7.62]
pi = [0.57, 0.11, 0.22, 0.10]
# training points
training_points = [1, 0, 10, -5]

for mean, std, pi in zip(means, stds, pi):
    # use gaussian formula to generate curve
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    y = pi * np.exp(-((x - mean) ** 2) / (2 * std**2))
    plt.plot(x, y, linewidth=2, label="mean: {}, std: {}, pi: {}".format(mean, std, pi))

# plot training points
plt.scatter(training_points, np.zeros(len(training_points)) + 0.01, c="r", marker="x", label="training points")
plt.title("Gaussian Mixture Model")
plt.xlabel("x")
plt.ylabel("y")

# limit y axis to 1
axes = plt.gca()
axes.set_ylim([0, 1])


plt.legend()
plt.show()
