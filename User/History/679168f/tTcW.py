import pandas as pd
from sb3_contrib import AttentionPPO

from reward import reward_function

import gymnasium as gym
import gym_trading_env
import pickle

from sb3_policy import CustomCNN


def get_dataset(split=0.99, sin=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sin:
        train_df = pd.read_pickle("dataset/out/simple_dataset.pkl")
        test_df = pd.read_pickle("dataset/out/simple_dataset_val.pkl")
    else:
        with open("dataset/EURUSD.pkl", "rb") as f:
            df = pickle.load(f)
        train_df = df.iloc[: int(len(df) * split)]
        test_df = df.iloc[int(len(df) * split) :]
    return train_df, test_df


def train():
    train_df, val_df = get_dataset(split=0.9999)

    episode_length = 500  # not exactly ppo but yeah who cares

    env = gym.make(
        "TradingEnv",
        df=train_df,
        positions=[-3, -2, -1, 1, 2, 3],
        reward_function=reward_function,
        # trading_fees=0.00025,
        portfolio_initial_value=100000,
        windows=45,
        max_episode_duration=episode_length,
    )

    env = 


    model = AttentionPPO("MlpAttnPolicy", env, verbose=1, gae_lambda=0.97, gamma=0.99)
    model.learn(total_timesteps=25000)

    # state_dim = env.observation_space.shape[1]  # state space dimension
    # action_dim = env.action_space.n  # action space dimension (positions the agent can take)


if __name__ == "__main__":
    train()
