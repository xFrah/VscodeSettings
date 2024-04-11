# create normal curve
import numpy as np


X = np.linspace(-6, 6, 1024)
Y = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(X**2.0) / 2)

import matplotlib.pyplot as plt
plt.plot(X, Y, c = 'k')
plt.show()