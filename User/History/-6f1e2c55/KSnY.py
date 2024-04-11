import numpy as np
import matplotlib.pyplot as plt

mean = 50
std = 15

x = np.random.normal(mean, std, 1000)

# plot curve
plt.hist(x, bins=100)