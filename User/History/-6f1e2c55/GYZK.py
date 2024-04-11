import numpy as np
import matplotlib.pyplot as plt

mean = 50
std = 15

# plot normal curve
x = np.linspace(0, 100, 100)
y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2))

# y limit to 2 * maximum y
plt.ylim(0, 1.5 * np.max(y))

# plot dotted lines at z-scores of -1, 0, 1
# plt.axvline(x=mean - 2 * std, color="red", linestyle="--")
# plt.axvline(x=mean - std, color="orange", linestyle="--")
# plt.axvline(x=mean, color="green", linestyle="--")
# plt.axvline(x=mean + std, color="orange", linestyle="--")
# plt.axvline(x=mean + 2 * std, color="red", linestyle="--")

z = (73 - mean) / std

# plot dotted lines at z
plt.axvline(x=mean + z * std, color="red", linestyle="--")

plt.plot(x, y)
plt.show()
