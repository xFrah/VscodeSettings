# create normal curve
import numpy as np


X = np.linspace(-6, 6, 1024)
Y = 1. / np.sqrt(2. * np.pi) * np.exp(-X ** 2. / 2)