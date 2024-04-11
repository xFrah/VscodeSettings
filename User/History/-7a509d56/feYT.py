import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from GTrXL import StableTransformerXL


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        self.transformer = StableTransformerXL(
            d_input=observation_space.shape[-1],
            n_layers=6,
            n_heads=12,
            d_head_inner=256,
            d_ff_inner=2048,
        )

        self.memory = None  # for storing the memory of transformer

        # linear layer
        self.linear = torch.nn.Linear(45, features_dim)

    def forward(self, state):
        """Forward pass of the policy."""
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state["logits"], trans_state["memory"]

        return trans_state
