import numpy as np
import matplotlib.pyplot as plt

mean = 50
std = 15

# plot normal curve
x = np.linspace(0, 100, 100)
y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2))

# 

plt.plot(x, y)
plt.show()
