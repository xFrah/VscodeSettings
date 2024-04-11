import matplotlib.pyplot as plt
import numpy as np

# create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

datapoints = [(0.10, 5), (15, 6), (20, 7), (25, 8), (30, 9), (80, 10)]
parameters = [0.11, 4]
y1 = None
x = np.arange(-100, 200, 0.1)

# limit ax2 y axis
ax2.set_ylim([0, 1.5])
ax2.set_xlim([-50, 90])


# function to transform x to y using parameters
def f(x, parameters):
    return parameters[0] * x + parameters[1]


for i, (x_, y_) in enumerate(datapoints):
    print("Predicted: ", f(x_, parameters), "Actual: ", y_)
    # plot vertical line that goes from (x_, 0) to (x_, y_)
    # ax2.plot([x_, x_], [0, f(x_, parameters)], "g--", linewidth=2)
    y = 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((y_ - f(x, parameters)) ** 2) / (2 * 3**2))
    if y1 is None:
        y1 = y
    else:
        y1 *= y
    # if i == 2:
    ax2.plot(x, y)
    # plot point in red
    ax2.scatter(x_, 1 / (3 * np.sqrt(2 * np.pi)) * np.exp(-((y_ - f(x_, parameters)) ** 2) / (2 * 3**2)), color="red")

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
