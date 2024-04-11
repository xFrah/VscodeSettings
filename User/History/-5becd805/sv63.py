import numpy as np


class ForexEnv:
    def __init__(self, data):
        # Store the data
        self.data = data
        # data is a day of data, each row is a candle that represents 1 minute, each row has 6 columns:
        # 0: datetime
        # 1: open price
        # 2: high price
        # 3: low price
        # 4: close price
        # 5: volume
        self.n_steps = len(data)

        # Define the actions
        self.action_space = ["buy", "hold", "sell"]

        # Initialize variables
        self.reset()

    def reset(self):
        # Reset the environment to the start of the data
        self.current_step = 0
        self.current_position = None  # current position can be 'buy', 'hold', or 'sell'
        self.profit = 0.0

        # Return the first state
        return self.get_state()

    def step(self, action):
        # Get the current price (close price of current candle)
        current_price = self.data[self.current_step, 1]  # close price

        # Update profit based on the action
        if action == "buy" and self.current_position is None:
            self.current_position = current_price
        elif action == "sell" and self.current_position is not None:
            self.profit += current_price - self.current_position
            self.current_position = None

        # Move to the next time step
        self.current_step += 1

        # Calculate the reward (change in profit)
        reward = self.profit
        if self.current_position is not None:
            reward += current_price - self.current_position

        # Check if the episode (day) is done
        done = self.current_step == self.n_steps

        # Return the next state, the reward, and whether the episode is done
        return self.get_state(), reward, done

    def get_state(self):
        # For now, the state will just be the current candle
        # But you might want to include other information as well (e.g., the current profit, etc.)
        return self.data[self.current_step]

    def sample_action(self):
        # Return a random action
        return np.random.choice(self.action_space)
