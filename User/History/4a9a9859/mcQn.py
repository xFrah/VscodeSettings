import matplotlib.pyplot as plt
import numpy as np

# create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

datapoints = [(10, 5), (15, 5), (20, 5), (25, 5), (30, 5), (80, 10)]
parameters = [0.07, 4]
y1 = None
x = np.arange(-100, 200, 0.1)


# function to transform x to y using parameters
def f(x, parameters):
    return parameters[0] * x + parameters[1]


for i, (x_, y_) in enumerate(datapoints):
    print("Predicted: ", f(x_, parameters), "Actual: ", y_)
    y = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((y_ - f(x, parameters)) ** 2) / (2 * 3**2))
    if y1 is None:
        y1 = y
    else:
        y1 *= y
    if i == 0:
        ax2.plot(x, y * 40)

# plot
# get product of all gaussians

# plot line with parameters
ax2.plot(x, f(x, parameters), "b--", linewidth=2)

ax1.plot(x, y1, "r--", linewidth=2)

# plot scatterplot of datapoints in ax2
ax2.scatter([x[0] for x in datapoints], [x[1] for x in datapoints])

# get x of max y
max_x = x[np.argmax(y1)]
print("Max x: ", max_x)
print("Mean of datapoints: ", np.mean([x[0] for x in datapoints]))


plt.show()
