from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from simenv import ForexEnv
import pickle
from categorical_transformer import StockPredictionModel
import threading
import time
import random
import numpy as np


# This class creates a replay buffer object. A replay buffer is a kind of data structure 
# that stores past experiences that the agent can use for training.
class ReplayBuffer:
    # The constructor method for the replay buffer, it takes the capacity of the buffer as an argument.
    def __init__(self, capacity):
        self.capacity = capacity  # The maximum number of experiences the buffer can hold.
        self.buffer = []  # The list that actually holds the experiences.
        self.position = 0  # The index at which the next experience will be stored.

    # This method adds an experience to the buffer. An experience is a tuple of (state, action, reward, next_state, done).
    def push(self, state, action, reward, next_state, done):
        # If the buffer is not full, append an empty slot to the buffer.
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # Add the experience at the current position.
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # Update the position for the next experience to be stored.
        # If the buffer is full, the position will be set to 0, and the oldest experiences will start to be replaced.
        self.position = (self.position + 1) % self.capacity

    # This method returns a random sample of experiences from the buffer.
    def sample(self, batch_size):
        # Randomly select 'batch_size' number of experiences.
        batch = random.sample(self.buffer, batch_size)
        # Unpack the experiences and group all the states together, all the actions together, etc.
        # Then, convert each group to a numpy array.
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    # This magic method returns the current number of experiences in the buffer when calling 'len(replay_buffer)'.
    def __len__(self):
        return len(self.buffer)

# ! Huber loss is less sensitive to outliers, look into it
# ! Implement LR scheduling?


hyperparameters = {
    "epsilon_initial": 1.0,  # Set the initial epsilon value for the epsilon-greedy policy
    "epsilon_min": 0.001,  # Minimum exploration rate
    "gamma": 0.9,  # Discount factor
    "learning_rate_initial": 0.1,  # Initial learning rate
    "bid_ask_spread": 0.0003,  # 3 pips spread
    "replay_buffer_capacity": 50000,  # Replay buffer capacity
    "batch_size": 10000,  # Batch size
}


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
    data = pickle.load(f)

d_model = 7
output_dim = 3  # Number of actions
nhead = 1  # Number of heads in the multi-head attention layers
nhid = 32  # Number of hidden units in the feedforward network model
nlayers = 2  # Number of layers in the Transformer model
dropout = 0.1  # Dropout rate


# Define the Transformer-based model (the Q function)
# ninp, nhead, nhid, nlayers, noutput, dropout=0.5
model = StockPredictionModel(d_model, nhead, nlayers, output_dim)
optimizer = optim.Adam(model.parameters())
criterion = nn.HuberLoss()


# Initialize epsilon (for the epsilon-greedy policy)
epsilon = hyperparameters["epsilon_initial"]

# Initialize the replay buffer
replay_buffer = ReplayBuffer(hyperparameters["replay_buffer_capacity"])

stats = []
action_stats = [0, 0, 0]


class LiveChart:
    def __init__(self) -> None:
        self.data = []
        # start the thread update
        self.thread = threading.Thread(target=self.update)

    def update(self):
        # series is a list of points, take the fifth element of each point and plot it as if it was a y value
        y_values = [point[4] for point in self.data]
        # plot with matplotlib
        plt.clf()
        plt.plot(y_values)
        # now check if each point is a buy or sell by checking the seventh element of each point, and plot a vertical line red or green accordingly
        for i, point in enumerate(self.data):
            if point[6] == 0.0:
                plt.axvline(x=i, color="green")
            elif point[6] == 1.0:
                plt.axvline(x=i, color="red")

        plt.show(block=False)
        plt.pause(0.1)

    def push(self, data):
        self.data = data


chart = LiveChart()

# Training loop
for episode in range(len(data)):
    env = ForexEnv(data[episode])
    state = env.reset()

    for _ in tqdm(range(len(data[episode]))):
        # Code for action selection here
        if torch.rand(1) < epsilon:
            action = env.sample_action()  # Exploration
        else:
            tensor = data_to_tensors(state)
            q_values = model(tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)

        chart.push(next_state)

        # Store the transition in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # Only start learning process if enough experiences have been accumulated in the buffer
        if len(replay_buffer) > hyperparameters["batch_size"]:
            # Sample a batch of experiences from the replay buffer
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                hyperparameters["batch_size"])

            with torch.no_grad():
                tensor = data_to_tensors(batch_next_states)
                target_q_values = batch_rewards + \
                    hyperparameters["gamma"] * \
                    torch.max(model(tensor), dim=1)[0]

            tensor = data_to_tensors(batch_states)
            temp = model(tensor)
            predicted_q_values = temp.gather(
                1, torch.tensor(batch_actions).unsqueeze(1))

            loss = criterion(predicted_q_values, target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

        if done:
            break

    # Update epsilon (for the epsilon-greedy policy) to gradually decrease the amount of exploration over time.
    # here we use linear decay
    epsilon = epsilon - \
        (hyperparameters["epsilon_initial"] -
            hyperparameters["epsilon_min"])/len(data)

    print("Episode: " + str(episode) + " Profit: " +
          str(env.portfolio_value-env.initial_capital) + " Epsilon: " + str(epsilon))

    stats.append(env.portfolio_value-env.initial_capital)
plt.clf()
plt.plot(stats)
plt.show()
plt.savefig("stats.png")
