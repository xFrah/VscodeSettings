import numpy as np
import torch
from GTrXL import StableTransformerXL
from torch.distributions.normal import Normal

class GTrXLBasis(NNBase):
def __init__(self, num_inputs, recurrent=False, hidden_size=64):
    super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

    if recurrent:
        num_inputs = hidden_size

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                           constant_(x, 0), np.sqrt(2))

    self.actor = nn.Sequential(
        init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

    self.critic = nn.Sequential(
        init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

    self.critic_linear = init_(nn.Linear(hidden_size, 1))

    self.train()

def forward(self, inputs, rnn_hxs, masks):
    x = inputs

    if self.is_recurrent:
        x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

    hidden_critic = self.critic(x)
    hidden_actor = self.actor(x)

    return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class TransformerPolicy(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_transformer_layers=8,
        n_attn_heads=16,
        d_head_inner=256,
        d_ff_inner=2048,
    ):
        """
        Initialize a Policy with Transformer Encoder (GTrXL) as backbone.
        :param state_dim: (int) dimension of state
        :param act_dim: (int) dimension of action
        :param n_transformer_layers: (int) number of transformer layers
        :param n_attn_heads: (int) number of attention heads
        :param d_head_inner: (int) dimension of inner layer in attention head
        :param d_ff_inner: (int) dimension of inner layer in feed forward network
        NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        """
        super(TransformerPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.transformer = StableTransformerXL(
            d_input=state_dim,
            n_layers=n_transformer_layers,
            action_space=act_dim,
            n_heads=n_attn_heads,
            d_head_inner=d_head_inner,
            d_ff_inner=d_ff_inner,
        )

        self.memory = None  # for storing the memory of transformer

    def forward(self, state):
        """Forward pass of the policy."""
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state["logits"], trans_state["memory"]

        if self.act_dim != 1:
            # apply softmax to get probabilities
            layer_out = torch.softmax(trans_state, dim=-1)
        else:
            # apply tanh to get action
            layer_out = torch.tanh(trans_state)

        return layer_out


if __name__ == "__main__":
    states = torch.randn(20, 5, 80)  # seq_size, batch_size, dim - better if dim % 2 == 0
    print("=> Testing Policy")
    policy = TransformerPolicy(state_dim=states.shape[-1], act_dim=3)
    for i in range(10):
        act = policy(states)
        print("Action" + str(act))
