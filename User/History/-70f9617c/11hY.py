

import numpy as np


x = np.arange(0, 16)
y_1 = np.exp(-((x - mean_1) ** 2) / (2 * std_1**2)) / (std_1 * np.sqrt(2 * np.pi))