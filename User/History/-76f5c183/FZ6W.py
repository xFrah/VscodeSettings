from matplotlib import pyplot as plt
import numpy as np

m = np.array([[-4, 9, 4, 0, -3], [10, -4, 8, 0, 0]])
mean = np.array([1.2, 2.8])
# subtract mean from m
m = m - mean[:, np.newaxis]
print(m)
# print covariance matrix
print(m @ m.T / 5)


means = [1.80, 1.13, 1.05, 1.12]
variance = [46.81, 6.91, 3.31, 7.62]
pi = [0.57, 0.11, 0.22, 0.10]
# training points
training_points = [1, 0, 10, -5]

for mean, var, pi in zip(means, variance, pi):
    # use gaussian formula to generate curve
    x = np.linspace(mean - 3 * np.sqrt(var), mean + 3 * np.sqrt(var), 100)
    y = np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

    # plot curve
    plt.plot(x, y, label="mean = " + str(mean) + ", variance = " + str(var) + ", pi = " + str(pi))

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
