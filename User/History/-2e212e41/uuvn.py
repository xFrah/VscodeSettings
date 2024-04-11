import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_stock_price(start, end, amplitude=0.5, frequency=0.01, noise_level=0.03):
    """
    Generates a sinusoidal stock price with noise.

    :param start: Start of the time range
    :param end: End of the time range
    :param steps: Number of data points in the range
    :param amplitude: Amplitude of the sine wave
    :param frequency: Frequency of the sine wave
    :param noise_level: Level of noise to add to the sine wave
    :return: time and stock price arrays
    """
    time = np.linspace(start, end, end)
    price = amplitude * np.sin(frequency * time)
    noise = np.random.normal(0, noise_level, end)
    price_with_noise = price + noise + 0.7

    return time, price_with_noise


def generate_dataframe(prices):
    # generate dataframe with high, low, open, close prices and volume
    df = pd.DataFrame(prices, columns=["c"])

    # add high, low, open prices
    df["h"] = df["c"] + np.random.uniform(0, 1, len(df))
    df["low"] = df["c"] - np.random.uniform(0, 1, len(df))

    # open price of first row is close price of previous row, use shift
    df["o"] = df["c"].shift(1)

    # add volume
    df["v"] = np.random.uniform(0, 1, len(df))

    # add time, each row is 1 minute
    df["t"] = pd.date_range("2009-01-01", periods=len(df), freq="T")

    df = df.dropna()

    return df


# Example usage
time, price = generate_stock_price(0, 5000000)
df = generate_dataframe(price)

plt.plot(time, price)
plt.title("Synthetic Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
