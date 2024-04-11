# generate point with covariance 0.8

import time
import numpy as np
import matplotlib.pyplot as plt


# generate point with covariance 0.8
def generate_point():
    x = np.random.normal(0, 1)
    y = 0.8 * x + np.random.normal(0, 1 - 0.8**2)
    return [x, y]


# generate 100 points
points = [generate_point() for i in range(100)]

# get inverse of covariance matrix of transformed points
cov = np.cov([p[0] for p in points], [p[1] for p in points])

# get first eigenvector
eig_val, eig_vec = np.linalg.eig(cov)
eig_vec = eig_vec[:, np.argmax(eig_val)]

# get unit vector with angle 43 degrees and norm 1
angle = 130
eig_vec = np.array([np.cos(np.pi / 180 * angle), np.sin(np.pi / 180 * angle)])


vec2 = cov @ eig_vec  # this just scales the eigenvector by the variance/eigenvalue in the direction of the eigenvector

# print magnitude of eigenvector
print("Eigenvector norm: ", np.linalg.norm(eig_vec))
print("Eigenvector angle: ", np.arctan2(eig_vec[1], eig_vec[0]) * 180 / np.pi)
print("Eigenvalue: ", eig_val[np.argmax(eig_val)])

# the dot product of the eigenvector and the covariance matrix times the eigenvector should be the eigenvalue because:
# a dot product between two vectors on the same direction is the product of their magnitudes (eig_vec.transpose() @ (cov @ eig_vec))
# since the eigenvector norm is 1, and cov @ eig_vec is the eigenvector scaled by the variance/eigenvalue,
# the dot product should be the eigenvalue
variance = eig_vec.transpose() @ cov @ eig_vec

print("Variance along first eigenvector", variance, "\n")

# project points onto eigenvector
transformed_points = [eig_vec.transpose() @ p for p in points]

# print magnitude of eigenvector
print("Σv norm: ", np.linalg.norm(vec2))
print("Σv angle: ", np.arctan2(vec2[1], vec2[0]) * 180 / np.pi, "\n")

print("vTΣv: ", eig_vec.transpose() @ cov @ eig_vec)
print("All this shit just to get the eigenvalue? Or the variance in the direction of the first eigenvector.")

while True:
    eig_vec = np.array([np.cos(np.pi / 180 * angle), np.sin(np.pi / 180 * angle)])

    vec2 = cov @ eig_vec  # this just scales the eigenvector by the variance/eigenvalue in the direction of the eigenvector

    # print magnitude of eigenvector
    print("Eigenvector norm: ", np.linalg.norm(eig_vec))
    print("Eigenvector angle: ", np.arctan2(eig_vec[1], eig_vec[0]) * 180 / np.pi)
    print("Eigenvalue: ", eig_val[np.argmax(eig_val)])

    # the dot product of the eigenvector and the covariance matrix times the eigenvector should be the eigenvalue because:
    # a dot product between two vectors on the same direction is the product of their magnitudes (eig_vec.transpose() @ (cov @ eig_vec))
    # since the eigenvector norm is 1, and cov @ eig_vec is the eigenvector scaled by the variance/eigenvalue,
    # the dot product should be the eigenvalue
    variance = eig_vec.transpose() @ cov @ eig_vec

    print("Variance along first eigenvector", variance, "\n")

    # project points onto eigenvector
    transformed_points = [eig_vec.transpose() @ p for p in points]

    # print magnitude of eigenvector
    print("Σv norm: ", np.linalg.norm(vec2))
    print("Σv angle: ", np.arctan2(vec2[1], vec2[0]) * 180 / np.pi, "\n")

    print("vTΣv: ", eig_vec.transpose() @ cov @ eig_vec)
    print("All this shit just to get the eigenvalue? Or the variance in the direction of the first eigenvector.")

    # plot points
    plt.scatter([p[0] for p in points], [p[1] for p in points], color="red")
    # plt.scatter([p for p in transformed_points], [0 for p in transformed_points], color="blue")
    # plot eigenvector
    plt.quiver(0, 0, vec2[0], vec2[1], color="green", scale=1)
    # plot eig_vec
    plt.quiver(0, 0, eig_vec[0], eig_vec[1], color="yellow", scale=1)

    plt.show(block=False)

    plt.pause(0.2)
    angle += 1
