import os
import numpy as np
import pandas as pd
import random as rd


class StockTradingEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99, beg_idx=0, end_idx=1113):
        self.df_pwd = r"C:\Users\fdimo\Downloads\China_A_shares.pandas.dataframe"
        self.npz_pwd = "./China_A_shares.numpy.npz"

        self.close_ary, self.tech_ary = self.load_data_from_disk()
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]
        print(f"| StockTradingEnv: close_ary.shape {self.close_ary.shape}")
        print(f"| StockTradingEnv: tech_ary.shape {self.tech_ary.shape}")

        self.max_stock = max_stock
        self.buy_cost_rate = 1 + buy_cost_pct
        self.sell_cost_rate = 1 - sell_cost_pct
        self.initial_amount = initial_amount
        self.gamma = gamma

        # reset()
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = 0
        self.if_random_reset = True

        self.amount = None
        self.shares = None
        self.shares_num = self.close_ary.shape[1]
        amount_dim = 1

        # environment information
        self.env_name = "StockTradingEnv-v2"
        self.state_dim = self.shares_num + self.close_ary.shape[1] + self.tech_ary.shape[1] + amount_dim
        self.action_dim = self.shares_num
        self.if_discrete = False
        self.max_step = len(self.close_ary)

    def reset(self):
        self.day = 0
        if self.if_random_reset:
            self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
            self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2**6).astype(int)
        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(self.shares_num, dtype=np.float32)

        self.rewards = []
        self.total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        return self.get_state()

    def get_state(self):
        state = np.hstack(
            (
                np.array(self.amount * 2**-16),
                self.shares * 2**-9,
                self.close_ary[self.day] * 2**-7,
                self.tech_ary[self.day] * 2**-6,
            )
        )
        return state

    def step(self, action):
        self.day += 1

        action = action.copy()
        action[(-0.1 < action) & (action < 0.1)] = 0
        action_int = (action * self.max_stock).astype(int)
        # actions initially is scaled between -1 and 1
        # convert into integer because we can't buy fraction of shares

        for index in range(self.action_dim):
            stock_action = action_int[index]
            adj_close_price = self.close_ary[self.day, index]  # `adjcp` denotes adjusted close price
            if stock_action > 0:  # buy_stock
                delta_stock = min(self.amount // adj_close_price, stock_action)
                self.amount -= adj_close_price * delta_stock * self.buy_cost_rate
                self.shares[index] += delta_stock
            elif self.shares[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.shares[index])
                self.amount += adj_close_price * delta_stock * self.sell_cost_rate
                self.shares[index] -= delta_stock

        state = self.get_state()

        total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        reward = (total_asset - self.total_asset) * 2**-6
        self.rewards.append(reward)
        self.total_asset = total_asset

        done = self.day == self.max_step - 1
        if done:
            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = total_asset / self.initial_amount
        return state, reward, done, {}

    def load_data_from_disk(self, tech_id_list=None):
        tech_id_list = (
            [
                "macd",
                "boll_ub",
                "boll_lb",
                "rsi_30",
                "cci_30",
                "dx_30",
                "close_30_sma",
                "close_60_sma",
            ]
            if tech_id_list is None
            else tech_id_list
        )

        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict["close_ary"]
            tech_ary = ary_dict["tech_ary"]
        elif os.path.exists(self.df_pwd):  # convert pandas.DataFrame to numpy.array
            df = pd.read_pickle(self.df_pwd)

            tech_ary = []
            close_ary = []
            df_len = len(df.index.unique())  # df_len = max_step
            for day in range(df_len):
                item = df.loc[day]

                tech_items = [item[tech].values.tolist() for tech in tech_id_list]
                tech_items_flatten = sum(tech_items, [])
                tech_ary.append(tech_items_flatten)

                close_ary.append(item.close)

            close_ary = np.array(close_ary)
            tech_ary = np.array(tech_ary)

            np.savez_compressed(
                self.npz_pwd,
                close_ary=close_ary,
                tech_ary=tech_ary,
            )
        else:
            error_str = (
                f"| StockTradingEnv need {self.df_pwd} or {self.npz_pwd}"
                f"  download the following file and save in `.`"
                f"  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.pandas.dataframe (2.1MB)"
            )
            raise FileNotFoundError(error_str)
        return close_ary, tech_ary


asd = StockTradingEnv()
asd.reset()