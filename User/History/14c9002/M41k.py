# we are computing and plotting the area between two z-scores.

from math import sqrt
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


mean = 100
std = sqrt(15) / sqrt(25)

print(mean, std)

a = 0.05

z1 = (98 - mean) / std
z2 = (102 - mean) / std

print(z1, z2)

print(f"Z1: {z1}")
print(f"Z2: {z2}")

print(st.norm.cdf(z2) - st.norm.cdf(z1))

print(f"Area 1: {st.norm.cdf(z1)}")
print(f"Area 2: {st.norm.cdf(z2)}")

# plot normal curve
x = np.linspace(95, 105, 1000)
y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2))

# y limit to 1.5 * maximum y
plt.ylim(0, 1.5 * np.max(y))

# shade area under curve from z1 to z2
plt.fill_between(x, 0, y, where=(x <= mean + z2 * std) & (x >= mean + z1 * std), color="red", alpha=0.5)

plt.plot(x, y)
plt.show()