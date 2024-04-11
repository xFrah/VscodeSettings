def trade_completion(history: dict, e=2, w=0.00025):
    RPnL = (history["portfolio_valuation", -1] - history["portfolio_valuation", 0]) / history["portfolio_valuation", 0]

    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL
