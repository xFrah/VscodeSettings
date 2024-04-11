# scatter points in 1d

import numpy as np
import matplotlib.pyplot as plt

# generate 1d point cloud with 20 points

mean = 0
x = np.random.normal(mean, 1, 20)

# plot
plt.plot(x, np.zeros_like(x), "x")
plt.show()
