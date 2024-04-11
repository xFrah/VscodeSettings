# scatter 10 correlated points with a given correlation coefficient

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# generate 10 correlated points
x = [5.0, 2.0, 3.0, 7.0, 3.0]
y = [25.0, 8.0, 7.0, 30.0, 10.0]

# compute standard deviations
x_std = np.std(x)
y_std = np.std(y)

# plot horizontal line at mean of y
plt.axhline(np.mean(y), color="blue", linewidth=1.5)

# plot vertical line at mean of x
plt.axvline(np.mean(x), color="orange", linewidth=1.5)

mean_x = np.mean(x)
mean_y = np.mean(y)

print(f"Mean of x: {mean_x}")
print(f"Mean of y: {mean_y}")

print(5.125 )

acc = 0

for x_, y_ in zip(x, y):
    t = (y_ - mean_y) * (x_ - mean_x)
    acc += t
    # plot line that goes from the first point to the horizontal line at the mean of y
    plt.plot([x_, x_], [y_, mean_y], color="red" if t < 0 else "green", linestyle="dashed", linewidth=1)

    # for the same point plot a line that goes from the point to the vertical line at the mean of x
    plt.plot([x_, mean_x], [y_, y_], color="red" if t < 0 else "green", linestyle="dashed", linewidth=1)

print(f"Covariance of x and y: {acc}")

r = acc / (x_std * y_std)
m = r * (y_std / x_std)


print(f"Standard deviation of x: {x_std}")
print(f"Standard deviation of y: {y_std}")
print(f"Product of standard deviations: {x_std * y_std}")

# plot the points
plt.scatter(x, y)

# plot regression line
plt.plot([0, 10], [0, m * 10], color="red")

# plot the regression line
# plt.plot(x, slope * x + intercept, color="red")

# show the plot
plt.show()
