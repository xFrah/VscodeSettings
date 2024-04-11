# scatter correlated 2d point cloud

import numpy as np
import matplotlib.pyplot as plt

# generate correlated 2d point cloud
mean = [-2, 3]
cov = [[1, -0.7], [-0.7, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T

# get covariance matrix
cov = np.cov(x, y)
print(cov)

# get eigenvalues and eigenvectors
eig_val, eig_vec = np.linalg.eig(cov)

# transform data to eigenspace
data = np.array([x, y])
# subtract mean
data[0] -= mean[0]
data[1] -= mean[1]

data = np.dot(eig_vec, data)

# create multiple plots
fig, ax = plt.subplots(1, 2)

# plot eigenvectors starting from mean
# plot
ax[0].plot(x, y, "x")
ax[0].plot([mean[0], mean[0] + eig_vec[0, 0] * eig_val[0]], [mean[1], mean[1] + eig_vec[1, 0] * eig_val[0]], color="red")
ax[0].plot([mean[0], mean[0] + eig_vec[0, 1] * eig_val[1]], [mean[1], mean[1] + eig_vec[1, 1] * eig_val[1]], color="red")

# plot transformed data
ax[1].plot(data[0], data[1], "x")


# put axis on equal scale
ax[0].set_aspect("equal", adjustable="box")
ax[1].set_aspect("equal", adjustable="box")

# set limits
ax[0].set_xlim(-8, 8)
ax[0].set_ylim(-8, 8)
ax[1].set_xlim(-8, 8)
ax[1].set_ylim(-8, 8)

plt.show()
