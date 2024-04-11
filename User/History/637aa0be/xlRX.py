import numpy as np
from matplotlib import pyplot as plt

probability = [0.25, 0.01, 0.4, 0.01, 0.165, 0.15, 0.02, 0.5]
labels = [1, -1, -1, -1, -1, 1, -1, +1]


def get_roc_point(probability, labels, threshold):
    tpr = 0
    fpr = 0
    for p, l in zip(probability, labels):
        if p >= threshold:
            if l == 1:
                tpr += 1
            else:
                fpr += 1
    return tpr, fpr


points = [get_roc_point(probability, labels, t) for t in np.arange(0, 1, 0.01)]
