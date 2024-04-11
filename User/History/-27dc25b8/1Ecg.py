def trade_completion(history: dict, e=2, w=0.0001):
    RPnL = 0  # agent’s realized PnL at time step t

    tc: float
    if RPnL >= e * w:
        return 1
    elif RPnL <= -w:
        return -1
    else:
        return RPnL
