from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


means = [1.80, 1.13, 1.05, 1.12]
stds = [46.81, 6.91, 3.31, 7.62]
pi = [0.57, 0.11, 0.22, 0.10]

for mean, std, pi in zip(means, stds, pi):
    # use gaussian formula to generate curve
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    y = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    plt.plot(x, y, linewidth=2, label='mean: {}, std: {}, pi: {}'.format(mean, std, pi))

plt.legend()
plt.show()