import datetime
import sys
import pandas as pd
from collections import deque
import os
import shutil


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
        self.prices = pd.read_csv(f"data/prices/input_pricedata/{self.symbol}_M15.csv")

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


def merge_prices_with_news(prices, news_path):
    prices["News"] = None  # add column for news

    news = pd.read_csv(news_path, parse_dates=True)
    news_list = news.values.tolist()

    # reverse list (to pop from the end and save time later)
    news_list_reverse = news_list[::-1]
    prices_list = prices.values.tolist()  # convert to list to iterate

    prices_len = len(prices_list)
    news_count = 0
    for i in range(prices_len):
        date_max = datetime.datetime.strptime(prices_list[i][0], "%Y-%m-%d %H:%M")
        date_min = date_max - datetime.timedelta(minutes=15)
        relevant_news = []
        while len(news_list_reverse) > 0:
            # pop from the end so it's O(1) instead of O(n) (we reversed the list)
            news_item = news_list_reverse.pop()
            news_date = datetime.datetime.strptime(news_item[1], "%Y-%m-%d %H:%M:%S")
            if news_date > date_min and news_date <= date_max:
                relevant_news.append(news_item[0])
            elif news_date < date_min:
                # article is too old, we can stop searching for this price item
                # we don't put it back because it's too old
                break
            elif news_date > date_max:
                # article is too new, we can put it back and stop searching for this price item
                news_list_reverse.append(news_item)
                break
        relevant_news_string = "<SEP>".join(relevant_news)
        # remove cr,lf,etc
        relevant_news_string = relevant_news_string.replace("\r", "").replace("\n", "").replace("\t", "")
        # add to prices
        prices.at[i, "News"] = relevant_news_string
        if len(relevant_news) > 0:
            news_count += 1

        if i % 25000 == 0:
            print(f"Progress: {i}/{prices_len}")

    print(f"Points with news/total): {news_count}/{prices_len}")


def make_dataset(symbol):
    if not os.path.exists("data/datasets"):  # final folder
        os.makedirs("data/datasets")

    print("Analyzing prices data...")
    # the first 50 rows are incomplete
    df = PriceDataCalculator(symbol).prices[50:]
    print("Merging prices with news...")
    merge_prices_with_news(df, "data/news/forexlive.csv")
    print("Saving to csv file...")
    df.to_csv(f"data/datasets/{symbol}_M15.csv", index=False)

    print("DONE!")


if __name__ == "__main__":
    make_dataset("EURUSD")
