import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
        n_transformer_layers: int,
        n_attn_heads: int,
        d_head_inner: int,
        d_ff_inner: int,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        self.transformer = StableTransformerXL(
            d_input=state_dim,
            n_layers=n_transformer_layers,
            n_heads=n_attn_heads,
            d_head_inner=d_head_inner,
            d_ff_inner=d_ff_inner,
        )

        self.memory = None  # for storing the memory of transformer

    def forward(self, state):
        """Forward pass of the policy."""
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state["logits"], trans_state["memory"]
        return trans_state
