import numpy as np
from scipy.stats import entropy


def entropy_(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def trade_completion(history: dict, e=2, w=0.00025):
    RPnL = (history["portfolio_valuation", -1] - history["portfolio_valuation", 0]) / history["portfolio_valuation", 0]

    e = entropy_(history["position"][-45:])

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL


a = [-3] * 9 + [-1.5] * 9 + [0] * 9 + [1.5] * 9 + [3] * 9
print(entropy_(a))
