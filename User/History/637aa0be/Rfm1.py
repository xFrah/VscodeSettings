import numpy as np
from matplotlib import pyplot as plt

probability = [0.25, 0.01, 0.4, 0.01, 0.165, 0.15, 0.02, 0.5]
labels = [1, -1, -1, -1, -1, 1, -1, +1]
threhsolds = [1.5, 0.5, 0.3, 0.25, 0.165, 0.15, 0.02, 0.01]


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


points = [get_roc_point(probability, labels, t) for t in threhsolds]

# plot ROC curve
plt.figure()
plt.plot([x[1] / 5 for x in points], [x[0] / 3 for x in points])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()