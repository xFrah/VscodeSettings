from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


means = [1.80, 1.13, 1.05, 1.12]
stds = [46.81, 6.91, 3.31, 7.62]
pi = [0.57, 0.11, 0.22, 0.10]

for mean, std, pi in zip(means, stds, pi):
    # create gaussian
    gaussian = np.random.normal(mean, std, 1000)
    # plot gaussian as line
    plt.plot(gaussian, np.zeros_like(gaussian) + pi, 'x')

plt.show()

