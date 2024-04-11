import numpy as np
import matplotlib.pyplot as plt

mean_1 = 3
std_1 = 2
mean_2 = 10
std_2 = 2

x = np.arange(0, 16)
y_1 = np.exp(-((x - mean_1) ** 2) / (2 * std_1**2)) / (std_1 * np.sqrt(2 * np.pi))
y_2 = np.exp(-((x - mean_2) ** 2) / (2 * std_2**2)) / (std_2 * np.sqrt(2 * np.pi))

# get surprise for distribution 1
surprise_2 = np.log(y_2)

# plot 2 bar charts, one beside the other

plt.bar(x, surprise_2, color="blue")
plt.bar(x, y_1 * 40, color="red", width=0.5)
plt.bar(x, y_2 * 40, color="blue", width=0.1)
plt.show()
