import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.001)
y = np.arange(-10, 10, 0.001)
point = (0, 0)
parameters = [1.5, 3]


def f(x, parameters):
    return parameters[0] * x + parameters[1]


mean = f(point[0], parameters)


likelihood = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((point[1] - mean) ** 2) / (2 * 3**2))

prediction_gaussian = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((y - mean) ** 2) / (2 * 3**2))
plt.plot(prediction_gaussian, x)
plt.scatter(likelihood, point[1], color="red")
plt.scatter(point[0], point[1], color="green")
plt.scatter(point[0], mean, color="blue")
# plot horizontal line that goes from green point to red likelihood point
plt.plot([point[0], likelihood], [point[1], point[1]], "--", linewidth=2)
# plot horintolal with hline


plt.show()
