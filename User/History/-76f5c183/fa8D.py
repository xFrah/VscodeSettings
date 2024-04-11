from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


means = [1.80, 1.13, 1.05, 1.12]
stds = [46.81, 6.91, 3.31, 7.62]
pi = [0.57, 0.11, 0.22, 0.10]

# craete gmm
gmm = GaussianMixture(n_components=4)
gmm.means_ = np.array(means).reshape(-1, 1)
gmm.covariances_ = np.array(stds).reshape(-1, 1, 1) ** 2
gmm.weights_ = np.array(pi)

# plot gmm
x = np.linspace(-100, 100, 1000)
y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
plt.plot(x, y)
plt.show()
