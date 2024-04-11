# we are computing and plotting the area between two z-scores.

from math import sqrt
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


mean = 0.88
std = sqrt((mean * (1 - mean)) / 100)

# test statistic
test_stat = (0.99 - mean) / std
print(f"Test statistic: {test_stat:.4f}")

print(mean, std)

alpha = 0.05

# z1 = (98 - mean) / std
# z2 = (102 - mean) / std

# print(z1, z2)

# print(f"Z1: {z1}")
# print(f"Z2: {z2}")

# print(st.norm.cdf(z2) - st.norm.cdf(z1))

# print(f"Area 1: {st.norm.cdf(z1)}")
# print(f"Area 2: {st.norm.cdf(z2)}")

# plot normal curve
x = np.linspace(0.35, 0.65, 1000)
y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2))

# y limit to 1.5 * maximum y
plt.ylim(0, 1.5 * np.max(y))

# shade area under curve from z1 to z2
# plt.fill_between(x, 0, y, where=(x <= mean + z2 * std) & (x >= mean + z1 * std), color="red", alpha=0.5)

# get z score given area under curve
z = st.norm.ppf(alpha / 2)
print(z)
print(mean - z * std)

# shade area under curve before -z and after z
plt.fill_between(x, 0, y, where=(x <= mean + z * std), color="red", alpha=0.5)
plt.fill_between(x, 0, y, where=(x >= mean - z * std), color="red", alpha=0.5, label=f"Significance level: {alpha:.4f}")

# plot vertical dotted line for sample mean
plt.axvline(x=mean + test_stat * std, linestyle="--", color="black", label=f"Sample proportion: {mean + test_stat * std:.4f}")

plt.legend()

plt.plot(x, y)
plt.show()
