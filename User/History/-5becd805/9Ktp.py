import numpy as np

action_dict = {"buy": 0, "hold": 1, "sell": 2}


class ForexEnv:
    def __init__(self, data):
        # Store the data
        self.data = data
        # data is a day of data, each element is a candle that represents 1 minute, each element is a dictionary of 7 elements:
        # 0: datetime
        # 1: open price
        # 2: high price
        # 3: low price
        # 4: close price
        # 5: volume
        # 6: action taken (buy, sell, hold)  # we change this at runtime, when we reach this step

        self.n_steps = len(data)

        # Define the actions
        self.action_space = [action_dict["buy"], action_dict["hold"], action_dict["sell"]]

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
        current_price = self.data[self.current_step][4]  # close price

        # set action in data
        self.data[self.current_step][6] = action

        # Update profit based on the action
        if action == action_dict["buy"] and self.current_position is None:
            self.current_position = current_price
        elif action == action_dict["sell"] and self.current_position is not None:
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
        # the state will be all the candles until now
        return self.data[: self.current_step + 1]

    def sample_action(self):
        # Return a random action
        return np.random.choice(self.action_space)
