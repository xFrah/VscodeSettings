# scatter 10 correlated points with a given correlation coefficient

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# generate 10 correlated points
x = [5.0, 2.0, 3.0, 7.0, 3.0]
y = [25.0, 8.0, 7.0, 30.0, 10.0]

# calculate the correlation coefficient
corr_coef = np.corrcoef(x, y)[0, 1]

# compute standard deviations
x_std = np.std(x)
y_std = np.std(y)

# compute the slope and intercept
slope = corr_coef * y_std / x_std
intercept = np.mean(y) - slope * np.mean(x)

# plot horizontal line at mean of y
plt.axhline(np.mean(y), color="blue", linewidth=1.5)

# plot vertical line at mean of x
plt.axvline(np.mean(x), color="orange", linewidth=1.5)

mean_x = np.mean(x)
mean_y = np.mean(y)

acc = 0

for x_, y_ in zip(x, y):
    t = (y_ - mean_y) * (x_ - mean_x)
    acc += t
    # plot line that goes from the first point to the horizontal line at the mean of y
    plt.plot([x_, x_], [y_, mean_y], color="red" if t < 0 else "green", linestyle="dashed", linewidth=1)

    # for the same point plot a line that goes from the point to the vertical line at the mean of x
    plt.plot([x_, mean_x], [y_, y_], color="red" if t < 0 else "green", linestyle="dashed", linewidth=1)

print(acc)

r = acc / (x_std * y_std)

m = r * (y_std / x_std)

print(r, corr_coef, corr_coef * (x_std * y_std))
print(f"Standard deviation of x: {x_std}")
print(f"Standard deviation of y: {y_std}")

# plot the points
plt.scatter(x, y)


# plot the regression line
# plt.plot(x, slope * x + intercept, color="red")

# show the plot
plt.show()
