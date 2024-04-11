import numpy as np
import pandas as pd
import polygon
from scipy import stats
import ta


def dl(symbol: str):
    forex_client = polygon.ForexClient("uy_C6zA5nsI2p9kX2I_bhsWGIAKEOyXL")  # for usual sync client
    data = forex_client.get_full_range_aggregate_bars(symbol, "2009-08-28", "2023-08-10")

    # convert data from dict list to pandas dataframe
    df = pd.DataFrame.from_dict(data)

    # convert t column to datetime (it's in milliseconds)
    df["t"] = pd.to_datetime(df["t"], unit="ms")

    # rename columns
    df = df.rename(
        columns={
            "t": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    # set date as index
    df = df.set_index("date")

    a = df["close"].min() / 2

    # subtract min/2 from close
    df["close"] = df["close"] - a
    df["high"] = df["high"] - a
    df["low"] = df["low"] - a
    df["open"] = df["open"] - a

    df["feature_close"] = df["close"]  # close price
    # df["feature_volume"] = df["volume"]  # volum
    # make feature_volume as the normalized volume, subtracting the mean and dividing by the standard deviation
    volume_mean = df["volume"].mean()
    volume_std = df["volume"].std()

    df["feature_volume"] = (df["volume"] - volume_mean) / volume_std

    df["feature_close_norm"] = np.log(df["close"] / df["close"].shift(1))  # log return
    df["feature_volume_norm"] = np.log(df["volume"] / df["volume"].shift(1))  # log return
    df.dropna(inplace=True)

    # add bollinger bands
    df["bb_high"] = ta.volatility.bollinger_hband(df["close"] + a, window=20)
    df["bb_low"] = ta.volatility.bollinger_lband(df["close"] + a, window=20)

    # add bb width
    df["feature_bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_high"]

    # add b%
    df["feature_bb_perc"] = ((df["close"] + a) - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    # add technical indicators
    df["feature_ema_short"] = ta.trend.ema_indicator(df["close"], window=25)
    df["feature_ema_long"] = ta.trend.ema_indicator(df["close"], window=60)
    df["feature_mfi"] = (
        ta.volume.MFIIndicator(
            df["high"] + a,
            df["low"] + a,
            df["close"] + a,
            df["volume"],
            window=14,
        ).money_flow_index()
        / 100
    )
    df["feature_rsi"] = ta.momentum.rsi(df["close"] + a, window=14) / 100
    df["feature_macd"] = ta.trend.macd(df["close"] + a, window_slow=26, window_fast=12)
    df["feature_cci"] = (
        ta.trend.cci(
            df["high"],
            df["low"],
            df["close"],
            window=20,
        )
        / 100
    )

    df.dropna(inplace=True)

    # add Ichimoku Hinko Hyo
    df["feature_ichimoku_a"] = ta.trend.ichimoku_a(df["high"], df["low"], window1=9, window2=26)
    df["feature_ichimoku_b"] = ta.trend.ichimoku_b(df["high"], df["low"], window2=52)
    df["feature_ichimoku_base_line"] = ta.trend.ichimoku_base_line(df["high"], df["low"], window1=9, window2=26)
    df["feature_ichimoku_conversion_line"] = ta.trend.ichimoku_conversion_line(df["high"], df["low"], window1=9, window2=26)

    df.dropna(inplace=True)

    # save to pickle
    df.to_pickle(f"dataset/out/{symbol}.pkl")


if __name__ == "__main__":
    dl("EURUSD")
