import numpy as np
from scipy.stats import entropy


def entropy_(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def trade_completion(history: dict, e=2, w=0.00025, max_e=1.6094379124341005):
    RPnL = (history["portfolio_valuation", -1] - history["portfolio_valuation", 0]) / history["portfolio_valuation", 0]

    ent = entropy_(history["position"][-45:]) / max_e

    if ent < 0.1:
        ent = 0.1

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL


a = [-3] * 0 + [-1.5] * 0 + [0] * 0 + [1.5] * 2 + [3] * 43
print(entropy_(a) / 1.6094379124341005)
