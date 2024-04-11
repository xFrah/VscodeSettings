import numpy as np
from scipy.stats import entropy


def entropy_(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def trade_completion(history: dict, e=2, w=0.00025):
    RPnL = (history["portfolio_valuation", -1] - history["portfolio_valuation", 0]) / history["portfolio_valuation", 0]

    _, counts = np.unique(labels, return_counts=True)
    if entropy(counts, base=base)

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL
