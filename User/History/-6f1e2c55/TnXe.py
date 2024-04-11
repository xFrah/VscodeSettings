import numpy as np
import matplotlib.pyplot as plt

mean = 50
std = 15

# crate two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# plot normal curve
x = np.linspace(0, 100, 100)
y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2))

# y limit to 1.5 * maximum y
ax1.set_ylim(0, 1.5 * np.max(y))
ax2.set_ylim(0, 1.5 * np.max(y))

# plot dotted lines at z-scores of -1, 0, 1
# plt.axvline(x=mean - 2 * std, color="red", linestyle="--")
# plt.axvline(x=mean - std, color="orange", linestyle="--")
# plt.axvline(x=mean, color="green", linestyle="--")
# plt.axvline(x=mean + std, color="orange", linestyle="--")
# plt.axvline(x=mean + 2 * std, color="red", linestyle="--")

z = -(73 - mean) / std

print(z)

# plot line that starts at mean + z*std and goes to the curve
ax1.plot(
    [mean + z * std, mean + z * std],
    [0, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((mean + z * std - mean) ** 2) / (2 * std**2))],
    color="red",
    linestyle="--",
)
ax2.plot(
    [mean + z * std, mean + z * std],
    [0, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((mean + z * std - mean) ** 2) / (2 * std**2))],
    color="red",
    linestyle="--",
)

# highlight area under curve from mean to z
ax1.fill_between(x, 0, y, where=(x <= mean + z * std), color="red", alpha=0.5)
ax2.fill_between(x, 0, y, where=(x <= mean + z * std), color="red", alpha=0.5)

ax1.plot(x, y)
ax2.plot(x, y)
plt.show()
