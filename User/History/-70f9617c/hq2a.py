import numpy as np

mean_1 = 3
std_1 = 2
mean_2 = 10
std_2 = 2

x = np.arange(0, 16)
y_1 = np.exp(-((x - mean_1) ** 2) / (2 * std_1**2)) / (std_1 * np.sqrt(2 * np.pi))

# get surprise for distribution 1
surprise_1 = np.log(y_1)

# plot 2 bar charts, one beside the other in 2 subplots
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(x, y_1, color="red")
ax2.bar(x, surprise_1, color="blue")
plt.show()
