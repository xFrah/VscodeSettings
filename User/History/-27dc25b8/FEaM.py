from scipy.stats import entropy
import numpy as np


def entropy_(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def reward_function(history: dict, e=2, w=0.0003, max_e=1.6094379124341005):
    RPnL = (history["portfolio_valuation", -1] - history["portfolio_valuation", -2]) / history["portfolio_valuation", -2]

    ent = entropy_(history["position"][-45:]) / max_e

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL if ent > 0.15 else -1


if __name__ == "__main__":
    entropy_([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
