import gym_trading_env
import numpy as np
import pandas as pd


def create_environment_with_dataset(dataset: str):
    # They need to be ordered by ascending date.
    # Index must be DatetimeIndex.
    # Your DataFrame needs to contain a close price labelled close for the environment
    # to run, and open, high, low, volume features respectively labelled
    # open , high , low , volume to perform renders
    dataset = pd.read_csv(dataset)
§