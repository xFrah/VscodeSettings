import datetime
import sys
import pandas as pd
from collections import deque
import os, shutil


class PriceDataCalculator:
    def __init__(self, symbol):
        self.symbol = symbol
        self.load_data()

    def load_data(self):
        self.load_prices()  # download all prices
        self.compute_bollinger_bands()  # compute bollinger bands
        self.compute_macd()  # compute macd
        self.compute_rsi()  # compute rsi
        self.compute_awesome_oscillator()  # compute awesome oscillator
        self.compute_stochastic_oscillator()  # compute stochastic oscillator
        self.compute_kmeans_clusters_support_resistance()  # compute support and resistance
        self.compute_accumulation_distribution()  # compute accumulation distribution
        self.compute_parabolic_sar()  # compute parabolic sar
        self.compute_fibonacci_retracement()  # compute retracement %

    def load_prices(self) -> pd.DataFrame:
        # read data from csv
        self.prices = pd.read_csv(f"data/prices/input_pricedata/{self.symbol}_M15.csv", index_col="Date", parse_dates=True)

    def compute_bollinger_bands(self):
        self.prices["Simple Moving Average (20)"] = self.prices["Close"].rolling(window=20).mean()
        self.prices["Standard Deviation (20)"] = self.prices["Close"].rolling(window=20).std()
        self.prices["Bollinger Band (20,2) Upper"] = self.prices["Simple Moving Average (20)"] + (
            self.prices["Standard Deviation (20)"] * 2
        )
        self.prices["Bollinger Band (20,2) Lower"] = self.prices["Simple Moving Average (20)"] - (
            self.prices["Standard Deviation (20)"] * 2
        )

    def compute_macd(self):
        self.prices["Exponential Moving Average (12)"] = self.prices["Close"].ewm(span=12).mean()
        self.prices["Exponential Moving Average (26)"] = self.prices["Close"].ewm(span=26).mean()
        self.prices["MACD Line"] = self.prices["Exponential Moving Average (12)"] - self.prices["Exponential Moving Average (26)"]
        self.prices["MACD Signal"] = self.prices["MACD Line"].ewm(span=9).mean()

    def compute_rsi(self):
        # difference from yesterday
        self.prices["Previous Day Change"] = self.prices["Close"].diff()
        # gain is the difference if positive, else 0
        self.prices["Previous Day Gain"] = self.prices["Previous Day Change"].mask(self.prices["Previous Day Change"] < 0, 0.0)
        # loss is the difference if negative, else 0
        self.prices["Previous Day Loss"] = -self.prices["Previous Day Change"].mask(self.prices["Previous Day Change"] > 0, -0.0)
        # average gain is the average of the gains for the last 14 days
        self.prices["Average Gain EWM (13,14)"] = self.prices["Previous Day Gain"].ewm(com=13, min_periods=14).mean()
        # average loss is the average of the losses for the last 14 days
        self.prices["Average Loss EWM (13,14)"] = self.prices["Previous Day Loss"].ewm(com=13, min_periods=14).mean()
        self.prices["Relative Strength"] = self.prices["Average Gain EWM (13,14)"] / self.prices["Average Loss EWM (13,14)"]
        self.prices["Relative Strength Index"] = 100 - (100 / (1 + self.prices["Relative Strength"]))

    def compute_awesome_oscillator(self):
        self.prices["Simple Moving Average (5)"] = self.prices["Close"].rolling(window=5).mean()
        self.prices["Simple Moving Average (34)"] = self.prices["Close"].rolling(window=34).mean()
        self.prices["Awesome Oscillator"] = self.prices["Simple Moving Average (5)"] - self.prices["Simple Moving Average (34)"]

    def compute_stochastic_oscillator(self):
        # this code computes the stochastic oscillator value for each day
        self.prices["Lowest Low (14)"] = self.prices["Low"].rolling(window=14).min()
        self.prices["Highest High (14)"] = self.prices["High"].rolling(window=14).max()
        self.prices["Stochastic Fast Signal (%K)"] = 100 * (
            (self.prices["Close"] - self.prices["Lowest Low (14)"]) / (self.prices["Highest High (14)"] - self.prices["Lowest Low (14)"])
        )
        self.prices["Stochastic Slow Signal (%D)"] = self.prices["Stochastic Fast Signal (%K)"].rolling(window=3).mean()

    def compute_kmeans_clusters_support_resistance(self):
        # todo: https://www.alpharithms.com/calculating-support-resistance-in-python-using-k-means-clustering-101517/
        pass

    def compute_accumulation_distribution(self):
        self.prices["Accumulation Distribution"] = (
            (self.prices["Close"] - self.prices["Low"]) - (self.prices["High"] - self.prices["Close"])
        ) / (self.prices["High"] - self.prices["Low"])

    def compute_parabolic_sar(self):
        # this code computes the parabolic sar value for each day
        indicator = PSAR()
        self.prices["PSAR"] = self.prices.apply(lambda x: indicator.calcPSAR(x["High"], x["Low"]), axis=1)
        # Add supporting data
        self.prices["Parabolic SAR Extreme Point"] = indicator.ep_list
        # trend is 1 = up, 0 = down
        self.prices["Parabolic SAR Trend"] = indicator.trend_list
        self.prices["Parabolic SAR Acceleration Factor"] = indicator.af_list

    def compute_fibonacci_retracement(self):
        # this code computes the fibonacci retracement value for each day based on the last 14 days
        self.prices["Fibonacci Retracement Level (14)"] = self.prices.apply(
            lambda x: self._compute_retracement(x["Highest High (14)"], x["Lowest Low (14)"], x["Previous Day Change"]), axis=1
        )

    def _compute_retracement(self, high, low, change):
        diff = high - low
        change_percentage = abs(change / diff)
        if change_percentage < 0.236:
            return 0
        if change_percentage < 0.382:
            return 1
        elif change_percentage < 0.5:
            return 2
        elif change_percentage < 0.618:
            return 3
        elif change_percentage < 1:
            return 4
        else:
            return 0

    def get_info_by_datetime(self, date: str) -> float:
        return self.prices.loc[date]


class PSAR:
    def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
        self.max_af = max_af
        self.init_af = init_af
        self.af = init_af
        self.af_step = af_step
        self.extreme_point = None
        self.high_price_trend = []
        self.low_price_trend = []
        self.high_price_window = deque(maxlen=2)
        self.low_price_window = deque(maxlen=2)

        # Lists to track results
        self.psar_list = []
        self.af_list = []
        self.ep_list = []
        self.high_list = []
        self.low_list = []
        self.trend_list = []
        self._num_days = 0

    def calcPSAR(self, high, low):
        if self._num_days >= 3:
            psar = self._calcPSAR()
        else:
            psar = self._initPSARVals(high, low)

        psar = self._updateCurrentVals(psar, high, low)
        self._num_days += 1

        return psar

    def _initPSARVals(self, high, low):
        if len(self.low_price_window) <= 1:
            self.trend = None
            self.extreme_point = high
            return None

        if self.high_price_window[0] < self.high_price_window[1]:
            self.trend = 1
            psar = min(self.low_price_window)
            self.extreme_point = max(self.high_price_window)
        else:
            self.trend = 0
            psar = max(self.high_price_window)
            self.extreme_point = min(self.low_price_window)

        return psar

    def _calcPSAR(self):
        prev_psar = self.psar_list[-1]
        if self.trend == 1:  # Up
            psar = prev_psar + self.af * (self.extreme_point - prev_psar)
            psar = min(psar, min(self.low_price_window))
        else:
            psar = prev_psar - self.af * (prev_psar - self.extreme_point)
            psar = max(psar, max(self.high_price_window))

        return psar

    def _updateCurrentVals(self, psar, high, low):
        if self.trend == 1:
            self.high_price_trend.append(high)
        elif self.trend == 0:
            self.low_price_trend.append(low)

        psar = self._trendReversal(psar, high, low)

        self.psar_list.append(psar)
        self.af_list.append(self.af)
        self.ep_list.append(self.extreme_point)
        self.high_list.append(high)
        self.low_list.append(low)
        self.high_price_window.append(high)
        self.low_price_window.append(low)
        self.trend_list.append(self.trend)

        return psar

    def _trendReversal(self, psar, high, low):
        # Checks for reversals
        reversal = False
        if self.trend == 1 and psar > low:
            self.trend = 0
            psar = max(self.high_price_trend)
            self.extreme_point = low
            reversal = True
        elif self.trend == 0 and psar < high:
            self.trend = 1
            psar = min(self.low_price_trend)
            self.extreme_point = high
            reversal = True

        if reversal:
            self.af = self.init_af
            self.high_price_trend.clear()
            self.low_price_trend.clear()
        else:
            if high > self.extreme_point and self.trend == 1:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = high
            elif low < self.extreme_point and self.trend == 0:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = low

        return psar


def make_dataset_prices(csvName: str, symbol) -> None:
    """
    This function creates a dataset from a csv file (ANALYZED).
    """
    # Read csv file
    data = pd.read_csv(csvName)
    sequences = []
    # Create sequences
    # Create sequences of 50 days each, using a sliding window of 1 day
    # sequences are tuples (x, y) where x is the sequence of 50 days and y is the price of the 51st day
    print("Price dataset progress: 0%")
    for i in range(len(data) - 50):
        x = data.iloc[i : i + 50]
        # turn x into an array
        x = x.values.tolist()
        change = data.iloc[i + 50]["Previous Day Change"]
        y = change
        sequences.append([x, y])
        if i % 1000 == 0:
            print("Price dataset progress: " + "{:.2f}".format(i / len(data) * 100) + "%")
    # Turn sequences into a pandas dataframe
    sequences = pd.DataFrame(sequences, columns=["x", "y"])
    # Save sequences to a file
    print("\n\n")
    return sequences


def make_dataset_prices_news(dataset_prices: str, dataset_news: str, symbol) -> None:
    print("Loading datasets...")
    dataset_prices = pd.read_csv(dataset_prices)
    dataset_news = pd.read_csv(dataset_news)

    dataset_prices_list = dataset_prices.values.tolist()
    dataset_news_list = dataset_news.values.tolist()

    final_dataset = []

    added_news = 0

    # convert the first item of dataset_prices_list to a list (it is a string)
    for i in range(len(dataset_prices_list)):
        # replace 'nan' with 0 in dataset_prices_list_1
        dataset_prices_list_1 = dataset_prices_list[i][0]
        dataset_prices_list_1 = dataset_prices_list_1.replace("nan", "0")
        prices = eval(dataset_prices_list_1)
        label = dataset_prices_list[i][1]
        # get date of the last bar of the prices (the 50th)
        date = prices[-1][0]
        date_datetime = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        useful_articles = []
        while len(dataset_news_list) > 0:
            news1 = dataset_news_list[0]
            date2 = news1[1]
            date2_datetime = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
            # check if date2_datetime is before date_datetime and if not break the loop
            if date2_datetime > date_datetime:
                # exit the while loop
                break
            if date2_datetime <= date_datetime and date2_datetime >= date_datetime - datetime.timedelta(minutes=15):
                # add the news to the prices
                useful_articles.append(str(news1[0]))
                added_news += 1
                dataset_news_list = dataset_news_list[1:]
            else:
                dataset_news_list = dataset_news_list[1:]
        if i % 1000 == 0:
            print("News merge progress: " + str(i) + "/" + str(len(dataset_prices_list)))
            print("Added news: " + str(added_news))
        # build the final list
        final_dataset.append([[prices, " ".join(useful_articles)], label])

    # Turn sequences into a pandas dataframe
    final_dataset = pd.DataFrame(final_dataset, columns=["x", "y"])
    # Save sequences to a file
    return final_dataset


def make_dataset(symbol):

    # make directories if not present: data/datasets, data/prices/working_pricedata
    import os

    if not os.path.exists("data/datasets"):  # final folder
        os.makedirs("data/datasets")
    if not os.path.exists("data/prices/working_pricedata"):
        os.makedirs("data/prices/working_pricedata")  # working folder

    pdc = PriceDataCalculator(symbol)
    pdc.prices.to_csv(f"data/prices/working_pricedata/{symbol}_M15_ANALYZED.csv", index=True)

    print("Created analyzed prices file, now creating dataset (only prices)")
    dataset_prices = make_dataset_prices(f"data/prices/working_pricedata/{symbol}_M15_ANALYZED.csv", symbol)
    dataset_prices.to_csv(f"data/prices/working_pricedata/{symbol}_M15_prices_dataset.csv", index=False)

    print("Merging price data and news...")
    final_dataset = make_dataset_prices_news(
        f"data/prices/working_pricedata/{symbol}_M15_prices_dataset.csv", "data/news/forexlive.csv", symbol
    )
    final_dataset.to_csv(f"data/datasets/{symbol}_dataset.csv", index=False)

    # Delete all files inside data/prices/working_pricedata (save space)
    folder = "data/prices/working_pricedata"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete folder
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    print("DONE!")


if __name__ == "__main__":
    make_dataset("GBPUSD")
