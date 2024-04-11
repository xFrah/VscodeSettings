from scipy.stats import entropy
import numpy as np


def entropy_(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def reward_function(history: dict, e=2, w=0.0003, max_e=1.6094379124341005):
    # get index of last different position
    index = -2
    for i in range(-2, -len(history["position"]) - 1, -1):
        index = i
        if history["position"][i] != history["position"][-1]:
            # print(f"Index {i}, position {history['position'][i]}")
            break
        # print(f"New index: {index}")
    RPnL = (history["portfolio_valuation", -1] - history["portfolio_valuation", index]) / history["portfolio_valuation", index]

    ent = entropy_(history["position"][-45:]) / max_e

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        # print(ent, RPnL)
        if ent < 0.3:
            return -0.5
        return RPnL
