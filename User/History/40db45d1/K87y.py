from collections import deque
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
from TradingGym import trading_env
from model.categorical_transformer import StockPredictionModel
import pandas as pd

hyperparameters = {
    "epsilon_initial": 1.0,  # Set the initial epsilon value for the epsilon-greedy policy
    "epsilon_min": 0.001,  # Minimum exploration rate
    "epsilon_decay": 0.999,  # Epsilon decay rate
    "gamma": 0.95,  # Discount factor
    "learning_rate_initial": 0.005,  # Initial learning rate
    "bid_ask_spread": 0.0006,  # 5 pips spread
    "d_model": 13,  # Number of characteristics
    "nhead": 1,  # Number of heads in the multi-head attention layers
    "nhid": 35,  # Number of hidden units in the feedforward network model
    "nlayers": 3,  # Number of layers in the Transformer model
    "output_dim": 3,  # Number of actions
    "replay_frequency": 100,  # How often to optimize the model
}


class Agent:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        # Define the Transformer-based model (the Q function)
        # ninp, nhead, nhid, nlayers, noutput, dropout=0.5
        self.model = StockPredictionModel(
            d_model=hyperparameters["d_model"],
            nhead=hyperparameters["nhead"],
            num_layers=hyperparameters["nlayers"],
            num_classes=hyperparameters["output_dim"],
        )
        self.optimizer = optim.Adamax(
            self.model.parameters(), lr=hyperparameters["learning_rate_initial"])
        self.criterion = nn.MSELoss()
        self.epsilon = hyperparameters["epsilon_initial"]
        # use cuda if available
        self.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.index = 0

    def choice_action(self, state):
        # your rule base conditon or your max Qvalue action or Policy Gradient action
        # action=0 -> do nothing
        # action=1 -> buy 1 share
        # action=2 -> sell 1 share
        # in this testing case we just build a simple random policy
        if np.random.rand(1) < self.epsilon or state is not None:
            return np.random.randint(3)
        else:
            tensor = data_to_tensors(state)
            tensor.to(self.device)
            return torch.argmax(self.model(tensor)).item()

    def optimize(self, state, next_state, reward, action):
        self.index += 1
        # Compute the target Q value. We want our current estimate of Q to move towards this target.
        # The target Q value is computed as the immediate reward plus the discounted future reward (Q value) of the next state.
        # Note that we use 'model(next_state).detach()' to make sure gradients are not backpropagated through the target.
        with torch.no_grad():
            tensor = data_to_tensors(next_state)
            tensor.to(self.device)
            target_q_value = reward + \
                hyperparameters["gamma"] * \
                torch.max(self.model(tensor).detach())

        # Compute the predicted Q value by passing the current state through our model and selecting the Q value of the chosen action.
        tensor = data_to_tensors(state)
        tensor.to(self.device)
        temp = self.model(tensor)
        predicted_q_value = temp[0][action]

        # Compute the loss as the mean squared error between the predicted Q value and the target Q value.
        # We will use this loss to optimize our model.
        loss = self.criterion(predicted_q_value, target_q_value)

        # print(f"{self.index}) Loss: {loss}, Predicted Q Value: {predicted_q_value}, Target Q Value: {target_q_value}, Reward: {reward}")

        # Optimize the model by backpropagating the loss and updating the model parameters.
        self.optimizer.zero_grad()  # First, we zero out the gradients from the previous step
        loss.backward()  # Then, we backpropagate the loss
        self.optimizer.step()  # Finally, we update the model parameters


def data_to_tensors(data):
    # data is a state, which is a list of 1 minute candles with d_model characteristics
    # we need to convert it to a tensor of shape (S, 1, E)
    # S is the number of minutes in the state
    # E is the number of characteristics
    # 1 is the batch size
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    tensor = tensor.transpose(0, 1)
    return tensor


# Load data from pickle file
# Shape (num_of_days, num_of_minutes, num_of_characteristics)
with open("data/datasets/EURUSD_M1.pkl", "rb") as f:
    # shape (num_of_days, num_of_minutes, num_of_characteristics)
    data = pickle.load(f)
    # remove the days with less than 1000 minutes
    data = [day for day in data if len(day) > 1000]
    # keep first 80% of the data for training
    data = data[: int(len(data) * 0.8)]


class ReplayBuffer:
    """
    A simple replay buffer implementation for storing and sampling experiences.
    """

    def __init__(self, capacity):
        """
        Initialize a new ReplayBuffer instance.

        Args:
            capacity (int): The maximum number of experiences that can be stored in the buffer.
                            Once the buffer is full, old experiences are discarded to make room for new ones.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Args:
            state: The state at the beginning of the transition.
            action: The action taken.
            reward: The reward received.
            next_state: The state at the end of the transition.
            done: A boolean flag indicating whether the episode ended after this transition.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A tuple of five lists: states, actions, rewards, next_states, and done_flags.
            Each list contains batch_size elements.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Return the number of experiences currently stored in the buffer.

        Returns:
            int: The number of experiences in the buffer.
        """
        return len(self.buffer)


epochs = 3
agent = Agent(hyperparameters)

# Create a replay buffer with a capacity of 10000 experiences
replay_buffer = ReplayBuffer(capacity=10000)
batch_size = 32

counter = hyperparameters["replay_frequency"] + 0


for epoch in range(epochs):
    print(f"Epoch {epoch}")
    # We start the learning process by running it for a fixed number of episodes.
    epoch_rewards = []
    for episode in tqdm(range(len(data))):
        episode_losses = []
        # turn the day's data into a pandas dataframe with columns columns=["Time", "Open", "High", "Low", "Close", "Volume", "Action"]
        data_day = data[episode]
        # data_day = [[i] + data_day for i in range(len(data_day))]

        df = pd.DataFrame(data_day, columns=[
                          "Time", "Open", "High", "Low", "Close", "Volume", "Action", "datetime"])
        df.drop(columns=["Time"], inplace=True)
        df.drop(columns=["Action"], inplace=True)

        # create column with index of row
        df["serial_number"] = df.index

        env = trading_env.make(
            env_id="training_v1",
            obs_data_len=60,
            step_len=1,  # step_len -> when call step rolling windows will + step_len
            df=df,
            # fee -> when each deal will pay the fee, set with your product
            fee=hyperparameters["bid_ask_spread"],
            max_position=1,  # max_position -> the max market position for you trading share
            # deal_col_name -> the column name for calculating the reward.
            deal_col_name="Close",
            feature_names=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ],  # feature_names -> list contain the feature columns to use in trading status.
        )

        state = env.reset()
        env.render()

        # For each step within one episode.
        for i in range(len(data_day) - 1):

            counter -= 1
            if counter == -hyperparameters["replay_frequency"]:
                counter = hyperparameters["replay_frequency"] + 0

            if counter < 0:
                # Sample a random batch of experiences from the replay buffer.
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(
                    batch_size)

                # Convert each batch of experiences into a tensor.
                state_batch = torch.tensor(
                    state_batch, dtype=torch.float32)  # ! optimize this
                action_batch = torch.tensor(action_batch, dtype=torch.int64)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
                next_state_batch = torch.tensor(
                    next_state_batch, dtype=torch.float32)
                done_batch = torch.tensor(done_batch, dtype=torch.float32)

                reward_batch = reward_batch.to(device="cuda")
                state_batch = state_batch.to(device="cuda")
                next_state_batch = next_state_batch.to(device="cuda")

                # Compute the target Q values for each experience in the batch.
                # The target Q value of each experience is the discounted future reward of the best action
                # from the next state, plus the immediate reward from the current state.
                with torch.no_grad():
                    target_q_batch = reward_batch + \
                        hyperparameters["gamma"] * \
                        torch.max(agent.model(next_state_batch), dim=1)[0]

                # Compute the predicted Q values for each experience in the batch.
                q_batch = agent.model(state_batch)
                action_batch = action_batch.to(device="cuda")

                # Select the Q value for the action that was taken.
                q_batch = q_batch.gather(
                    dim=1, index=action_batch.unsqueeze(1)).squeeze(1)

                # Compute the loss between the predicted Q values and the target Q values.
                loss = agent.criterion(q_batch, target_q_batch)

                # Optimize the model by backpropagating the loss and updating the model parameters.
                agent.optimizer.zero_grad()

                loss.backward()
                agent.optimizer.step()

                print(f"Buffer: {i})")

            else:
                # Select action via epsilon-greedy policy.
                # We either select a random action (exploration) or choose the action with the highest Q value (exploitation)
                action = agent.choice_action(state)

                # Execute the action in the environment to get the next state and the reward.
                # 'done' is a boolean indicating if the episode has ended.
                next_state, reward, done, info = env.step(action)

                # Add the experience to the replay buffer.
                replay_buffer.push(state, action, reward, next_state, done)

                # Compute the target Q value. We want our current estimate of Q to move towards this target.
                # The target Q value is computed as the immediate reward plus the discounted future reward (Q value) of the next state.
                # Note that we use 'model(next_state).detach()' to make sure gradients are not backpropagated through the target.
                with torch.no_grad():
                    tensor = data_to_tensors(next_state).to(device="cuda")
                    target_q_value = reward + \
                        hyperparameters["gamma"] * \
                        torch.max(agent.model(tensor).detach())

                # Compute the predicted Q value by passing the current state through our model and selecting the Q value of the chosen action.
                tensor = data_to_tensors(state).to(device="cuda")
                temp = agent.model(tensor)
                predicted_q_value = temp[0][action]

                # Compute the loss as the mean squared error between the predicted Q value and the target Q value.
                # We will use this loss to optimize our model.
                loss = agent.criterion(predicted_q_value, target_q_value)

                # print(f"{self.index}) Loss: {loss}, Predicted Q Value: {predicted_q_value}, Target Q Value: {target_q_value}, Reward: {reward}")

                # Optimize the model by backpropagating the loss and updating the model parameters.
                agent.optimizer.zero_grad()  # First, we zero out the gradients from the previous step
                loss.backward()  # Then, we backpropagate the loss
                agent.optimizer.step()  # Finally, we update the model parameters

                # Update the current state with the next state in preparation for the next step.
                state = next_state

                print(f"No-Buffer: {i})")

            # If the episode has ended (i.e., if 'done' is True), we break out of the loop for this episode.
            if done:
                break

            env.render()

        # Update epsilon (for the epsilon-greedy policy) to gradually decrease the amount of exploration over time.
        # here we use exponential decay
        # compute the new epsilon value
        epsilon_temp = agent.epsilon * \
            hyperparameters["epsilon_decay"] ** episode
        # clip the value to be at least epsilon_min
        agent.epsilon = max(hyperparameters["epsilon_min"], epsilon_temp)
        epoch_rewards.append(np.mean(env.reward_arr))
        env.deleteplots()
    print(f"\n\nEpoch {epoch}, Mean reward: {np.mean(epoch_rewards)}\n\n")
    # reset epsilon for each epoch but to 3/4 of the initial value
    agent.epsilon = hyperparameters["epsilon_initial"] * 3 / (4 * (epoch + 1))
    plt.clf()
    plt.plot(epoch_rewards)
    plt.savefig(f"rewards_epoch_{epoch}.png")

torch.save(agent.model.state_dict(), f"data/weights/epoch_{epoch}.pt")
