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

# plot dotted lines at z that stops at the normal curve
plt.plot([z, z], [0, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((z * std + mean - mean) ** 2) / (2 * std**2))], color="blue", linestyle="--")

plt.plot(x, y)
plt.show()
