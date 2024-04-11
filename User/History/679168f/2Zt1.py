import time
import pandas as pd
from stable_baselines3 import A2C

import numpy as np
import torch
from a2c_acktr import Policy
from reward import reward_function

import gymnasium as gym
import gym_trading_env
import pickle

from a2c_acktr import A2C_ACKTR

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

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space.n,
    )
    actor_critic.to(device)

    agent = A2C_ACKTR(
        actor_critic,
        value_loss_coef,
        entropy_coef,
        lr=lr,
        eps=eps,
        alpha=alpha,
        max_grad_norm=max_grad_norm,
    )

    rollouts = RolloutStorage(
        num_steps,
        num_processes,
        observation_space.shape,
        action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    for j in range(num_updates):

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step]
                )

            # Obser reward and next obs
            obs, reward, done, truncated, infos = env.step(action)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )


if __name__ == "__main__":
    train()
