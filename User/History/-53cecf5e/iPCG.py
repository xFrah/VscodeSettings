import numpy as np
import torch
from model.GTrXL import StableTransformerXL
from torch.distributions.normal import Normal


class TransformerGaussianPolicy(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_transformer_layers=4,
        n_attn_heads=3,
        d_head_inner=32,
        d_ff_inner=64,
    ):
        """
        Initialize a Gaussian Policy with Transformer Encoder (GTrXL) as backbone.
        :param state_dim: (int) dimension of state
        :param act_dim: (int) dimension of action
        :param n_transformer_layers: (int) number of transformer layers
        :param n_attn_heads: (int) number of attention heads
        :param d_head_inner: (int) dimension of inner layer in attention head
        :param d_ff_inner: (int) dimension of inner layer in feed forward network
        NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        """
        super(TransformerGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.transformer = StableTransformerXL(
            d_input=state_dim,
            n_layers=n_transformer_layers,
            n_heads=n_attn_heads,
            d_head_inner=d_head_inner,
            d_ff_inner=d_ff_inner,
        )

        self.memory = None  # for storing the memory of transformer

        self.head_sate_value = torch.nn.Linear(state_dim, 1)  # state value (head)
        self.head_act_mean = torch.nn.Linear(state_dim, act_dim)  # action mean (head)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)  # log standard deviation
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))  # as a parameter

        self.tanh = torch.nn.Tanh()  # tanh activation
        self.relu = torch.nn.ReLU()  # relu activation

    def _distribution(self, trans_state):
        """Returns the distribution of action given state."""
        mean = self.tanh(self.head_act_mean(trans_state))
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob_from_distribution(self, policy, action):
        """Returns log probability of given action."""
        return policy.log_prob(action).sum(axis=-1)

    def forward(self, state, action=None):
        """Forward pass of the policy."""
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state["logits"], trans_state["memory"]

        policy = self._distribution(trans_state)
        state_value = self.head_sate_value(trans_state)

        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a, state_value

    def step(self, state):
        """Returns action, log probability and state value for given state."""
        if state.shape[0] == self.state_dim:
            state = state.reshape(1, 1, -1)
        with torch.no_grad():
            trans_state = self.transformer(state, self.memory)
            trans_state, self.memory = trans_state["logits"], trans_state["memory"]

            policy = self._distribution(trans_state)
            action = policy.sample()
            logp_a = self._log_prob_from_distribution(policy, action)
            state_value = self.head_sate_value(trans_state)

        return action.numpy(), logp_a.numpy(), state_value.numpy()


if __name__ == "__main__":
    states = torch.randn(20, 5, 8)  # seq_size, batch_size, dim - better if dim % 2 == 0
    print("=> Testing Policy")
    policy = TransformerGaussianPolicy(state_dim=states.shape[-1], act_dim=4)
    for i in range(10):
        act = policy(states)
        action = act[0].sample()
        print(torch.isnan(action).any(), action.shape)
