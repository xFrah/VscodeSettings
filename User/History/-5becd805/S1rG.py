import random
import numpy as np
import pandas as pd
from TradingGym import trading_env

df = pd.read_hdf("dataset/SGXTW.h5", "STW")




state, reward, done, info = env.step(random.randrange(3))

# randow choice action and show the transaction detail
for i in range(500):
    print(i)
    state, reward, done, info = env.step(random.randrange(3))
    print(state, reward)
    env.render()
    if done:
        break
env.transaction_details
