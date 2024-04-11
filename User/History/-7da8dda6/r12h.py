# scatter correlated 2d point cloud

import numpy as np
import matplotlib.pyplot as plt

# generate correlated 2d point cloud
mean = [-2, 3]
cov = [
    [-0.8, 1],
    [1, -0.8],
]
x, y = np.random.multivariate_normal(mean, cov, 1000).T

# subtract mean from data
x -= mean[0]
y -= mean[1]

# get covariance matrix
cov = np.cov(x, y)
print(cov)

# get eigenvalues and eigenvectors
eig_val, eig_vec = np.linalg.eig(cov)

# transform data to eigenspace
data1 = np.array([x, y])
# print(data1)
# data = np.dot(eig_vec, data)

# multiply data with transposed eigenvectors
data1 = np.dot(eig_vec.T, data1)

# copt the eigenvectors into new variable and swap the two eigenvectors
eig_vec2 = np.copy(eig_vec)
eig_vec2[:, [0, 1]] = eig_vec2[:, [1, 0]]

# put basis vectors to new eigenvectors
data2 = np.dot(eig_vec.T, data1)

# print data2 covariance matrix
print(np.cov(data2))


# create multiple plots
fig, ax = plt.subplots(1, 3)

# plot eigenvectors starting from mean
# plot
ax[0].plot(x, y, "x")
ax[0].plot([0, eig_vec[0, 0] * eig_val[0]], [0, eig_vec[1, 0] * eig_val[0]], color="red")
ax[0].plot([0, eig_vec[0, 1] * eig_val[1]], [0, eig_vec[1, 1] * eig_val[1]], color="red")

# plot transformed data
ax[1].plot(data1[0], data1[1], "x")


ax[2].plot(data2[0], data2[1], "x")
ax[2].plot([0, eig_vec2[0, 0] * eig_val[0]], [0, eig_vec2[1, 0] * eig_val[0]], color="red")
ax[2].plot([0, eig_vec2[0, 1] * eig_val[1]], [0, eig_vec2[1, 1] * eig_val[1]], color="red")


# put axis on equal scale
ax[0].set_aspect("equal", adjustable="box")
ax[1].set_aspect("equal", adjustable="box")
ax[2].set_aspect("equal", adjustable="box")

# set limits
ax[0].set_xlim(-8, 8)
ax[0].set_ylim(-8, 8)
ax[1].set_xlim(-8, 8)
ax[1].set_ylim(-8, 8)
ax[2].set_xlim(-8, 8)
ax[2].set_ylim(-8, 8)

plt.show()
