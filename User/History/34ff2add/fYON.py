# make discrete distribution from 0 to 10
# and plot the distribution

import numpy as np
import matplotlib.pyplot as plt

mean_1 = 4
std_1 = 3
mean_2 = 11
std_2 = 3

# set y limit to 0.2
plt.ylim(0, 0.2)

# make discrete distribution
x_1 = np.arange(0, 16)
y_1 = np.exp(-(x_1 - mean_1)**2 / (2 * std_1**2)) / (std_1 * np.sqrt(2 * np.pi))

x_2 = np.arange(0, 16)
y_2 = np.exp(-(x_2 - mean_2)**2 / (2 * std_2**2)) / (std_2 * np.sqrt(2 * np.pi))

# plot the two distributions together, the first in blue and the second in red as a bar chart
plt.bar(x_1, y_1, color='blue')
plt.bar(x_2, y_2, color='red')
plt.show()
