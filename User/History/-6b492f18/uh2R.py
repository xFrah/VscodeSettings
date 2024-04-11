import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from tqdm import tqdm
from TransformerPolicy import TransformerPolicy

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = TransformerPolicy(state_dim, action_dim)
        # if has_continuous_action_space:
        #     self.actor = nn.Sequential(
        #         nn.Linear(state_dim, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, action_dim),
        #         nn.Tanh(),
        #     )
        # else:
        #     self.actor = nn.Sequential(
        #         nn.Linear(state_dim, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, action_dim),
        #         nn.Softmax(dim=-1),
        #     )
        # critic
        self.critic = TransformerPolicy(state_dim, 1)
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 1),
        # )

    def act(self, state):
        action_probs = self.actor(state).squeeze(0)
        # # turn action_probs to probability distribution
        # action_probs = nn.Softmax(dim=-1)(action_probs)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state).squeeze(0)
        # # turn action_probs to probability distribution
        # action_probs = nn.Softmax(dim=-1)(action_probs)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        entropy_coefficient=0.01,
        lambda_gae=0.95,
    ):
        self.entropy_coefficient = entropy_coefficient
        self.lambda_gae = lambda_gae
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, val=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        if not val:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        # Prepare value estimates, including the next value after the last state
        next_value = 0  # If the last state is terminal; otherwise, compute appropriately
        state_values = self.buffer.state_values + [next_value]

        # GAE
        advantages = []
        gae = 0
        for t in reversed(range(len(self.buffer.rewards))):
            delta = self.buffer.rewards[t] + self.gamma * state_values[t + 1] * (1 - self.buffer.is_terminals[t]) - state_values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - self.buffer.is_terminals[t]) * gae
            advantages.insert(0, gae)

        # Compute returns by adding the values to the advantages
        rewards = torch.tensor(advantages, dtype=torch.float32).to(device) + torch.tensor(state_values[:-1], dtype=torch.float32).to(device)

        # Normalizing the advantages
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=1)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coefficient * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
