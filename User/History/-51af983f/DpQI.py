import matplotlib.pyplot as plt
import numpy as np

# set limits to -10 and 10 for y and 0 and 0.4 for x
plt.ylim([-7, 13])
plt.xlim([-0.05, 0.43])

x = np.arange(-5, 10, 0.001)
y = np.arange(-5, 10, 0.001)
point = (0, 0)
point2 = (0.25, 5)
parameters = [25, 0.1]


def f(x, parameters):
    return parameters[0] * x + parameters[1]


mean = f(point[0], parameters)
mean2 = f(point2[0], parameters)


likelihood = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((point[1] - mean) ** 2) / (2 * 3**2))
prediction_gaussian = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((y - mean) ** 2) / (2 * 3**2))

likelihood2 = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((point2[1] - mean2) ** 2) / (2 * 3**2)) + point2[0]

prediction_gaussian2 = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((y - mean2) ** 2) / (2 * 3**2))

plt.plot(prediction_gaussian, x)
plt.plot(prediction_gaussian2 + point2[0], x)
plt.scatter(point[0], point[1], color="green")
plt.scatter(point[0], mean, color="blue")
plt.scatter(point2[0], mean2, color="blue")
# plot horizontal line that goes from green point to red likelihood point
plt.plot([point[0], likelihood], [point[1], point[1]], "g--", linewidth=2, label="Likelihood")
plt.plot([point2[0], likelihood2], [point2[1], point2[1]], "g--", linewidth=2)

# plot regression line
plt.plot(x, f(x, parameters), "b--", linewidth=2, label="Regression line")

# plot horintolal with hline
# plot point2 in green
plt.scatter(point2[0], point2[1], color="green")


plt.show()
