import numpy as np


means = [1.80, 1.13, 1.05, 1.12]
stds = [46.81, 6.91, 3.31, 7.62]
pi = [0.57, 0.11, 0.22, 0.10]

# generate gaussians with means, stds and pi, like a gmm
x = np.random.normal(means, stds, 1000)

# plot
plt.scatter(x, np.zeros_like(x))
plt.show()
