def trade_completion(history: dict, e=2, w=0.0003):
    RPnL = history["portfolio_valuation", -1] history["portfolio_valuation", -1]

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL
