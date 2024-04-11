# make discrete distribution from 0 to 10
# and plot the distribution

import numpy as np
import matplotlib.pyplot as plt

mean_1 = 6
std_1 = 3
mean_2 = 7
std_2 = 3

# set y limit to 0.2
# plt.ylim(0, 0.2)

# make discrete distribution
x = np.arange(0, 16)
y_1 = np.exp(-(x - mean_1)**2 / (2 * std_1**2)) / (std_1 * np.sqrt(2 * np.pi))
y_2 = np.exp(-(x - mean_2)**2 / (2 * std_2**2)) / (std_2 * np.sqrt(2 * np.pi))

# print expected values
print("Expected value of distribution 1: ", np.sum(x * y_1))
print("Expected value of distribution 2: ", np.sum(x * y_2))

# compute Kullback-Leibler divergence between the two distributions
kl_div = np.sum(y_1 * np.log(y_1 / y_2))
print("Kullback-Leibler divergence between the two distributions: ", kl_div)

# for every valule of x, compute surprise for distribution 1 and 2
surprise_1 = np.log(y_1)
surprise_2 = np.log(y_2)

# plot the two distributions together, the first in blue and the second in red as a bar chart
plt.bar(x, surprise_1, color='blue')
plt.bar(x, y_1, color='red', width=0.4)
plt.show()
