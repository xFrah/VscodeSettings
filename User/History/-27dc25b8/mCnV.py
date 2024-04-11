def trade_completion(history: dict, e=2, w=0.0001):
    RPnL = history["portfolio_valuation", -1]

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL
