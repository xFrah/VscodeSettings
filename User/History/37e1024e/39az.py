import os
import time
import numpy as np
import numpy.random as rd
import pandas as pd
import torch


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


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = [map(list, zip(*traj_list))]
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp


class Arguments:
    def __init__(self, agent, env_func=None, env_args=None):
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.env_args["env_num"]  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.env_args["max_step"]  # the max step of an episode
        self.env_name = self.env_args["env_name"]  # the env name. Be used to set 'cwd'.
        self.state_dim = self.env_args["state_dim"]  # vector dimension (feature number) of state
        self.action_dim = self.env_args["action_dim"]  # vector dimension (feature number) of action
        self.if_discrete = self.env_args["if_discrete"]  # discrete or continuous action space

        self.agent = agent  # DRL algorithm
        self.net_dim = 2**7  # the middle layer dimension of Fully Connected Network
        self.batch_size = 2**7  # num of transitions sampled from replay buffer.
        self.mid_layer_num = 1  # the middle layer number of Fully Connected Network
        self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
        self.if_use_old_traj = False  # save old data to splice and get a complete trajectory (for vector env)
        if self.if_off_policy:  # off-policy
            self.max_memo = 2**21  # capacity of replay buffer
            self.target_step = 2**10  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2**0  # collect target_step, then update network
        else:  # on-policy
            self.max_memo = 2**12  # capacity of replay buffer
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2**4  # collect target_step, then update network

        """Arguments for training"""
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2**0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2**-12  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2**-8  # 2 ** -8 ~= 5e-3

        """Arguments for device"""
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        """Arguments for evaluate"""
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        """Arguments for evaluate"""
        self.eval_gap = 2**7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2**4  # number of times that get episode return

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        """auto set cwd (current working directory)"""
        if self.cwd is None:
            self.cwd = f"./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}"

        """remove history"""
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == "y")
        elif self.if_remove:
            import shutil

            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find("PPO") == -1, name.find("A2C") == -1))  # if_off_policy


def train_agent():
    torch.set_grad_enabled(False)

    target_step = max_step * 4
    reward_scale = 2**-7
    learning_rate = 2**-14
    break_step = int(5e5)


    """init"""
    env = StockTradingEnv()

    agent = MyAgent()
    agent.states = [
        env.reset(),
    ]

    buffer = ReplayBufferList()

    """start training"""
    cwd = os.getcwd()
    break_step = break_step
    target_step = target_step
    del args

    start_time = time.time()
    total_step = 0
    save_gap = int(5e4)
    total_step_counter = -save_gap
    while True:
        trajectory = agent.explore_env(env, target_step)
        steps, r_exp = buffer.update_buffer((trajectory,))

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        total_step += steps

        if total_step_counter + save_gap < total_step:
            total_step_counter = total_step
            print(
                f"Step:{total_step:8.2e}  "
                f"ExpR:{r_exp:8.2f}  "
                f"Returns:{env.cumulative_returns:8.2f}  "
                f"ObjC:{logging_tuple[0]:8.2f}  "
                f"ObjA:{logging_tuple[1]:8.2f}  "
            )
            save_path = f"{cwd}/actor_{total_step:014.0f}_{time.time() - start_time:08.0f}_{r_exp:08.2f}.pth"
            torch.save(agent.act.state_dict(), save_path)

        if (total_step > break_step) or os.path.exists(f"{cwd}/stop"):
            # stop training when reach `break_step` or `mkdir cwd/stop`
            break

    print(f"| UsedTime: {time.time() - start_time:.0f} | SavedDir: {cwd}")


def get_episode_return_and_step(env, act) -> (float, int):  # [ElegantRL.2022.01.01]
    """
    Evaluate the actor (policy) network on testing environment.

    :param env: environment object in ElegantRL.
    :param act: Actor (policy) network.
    :return: episodic reward and number of steps needed.
    """
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    episode_step = None
    episode_return = 0.0  # sum of rewards in an episode
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, "cumulative_returns", episode_return)
    episode_step += 1
    return episode_return, episode_step


def load_torch_file(model, _path):
    state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)


"""train and evaluate"""


def run():
    import sys

    env = StockTradingEnv()

    env_args["beg_idx"] = 0  # training set
    env_args["end_idx"] = 834  # training set

    train_agent()


if __name__ == "__main__":
    run()
