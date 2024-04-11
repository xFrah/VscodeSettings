def trade_completion(history: dict, e=2, w=0.0001):
    rpnl = 0  # RP nL is the agent’s realized PnL at time step t 

    tc: float
    if rpnl >= e * w:
        return 1
    elif rpnl <= -w:
        return -1
    else:
        return rpnl
