import datetime
import pandas as pd
import os
import pickle
import numpy as np


def make_dataset(symbol: str, interval: int) -> None:
    print(f"Making dataset for {symbol} with interval {interval}...")
    # Read data from csv file
    points = pd.read_csv(
        f"data/prices/{symbol}_M{interval}.csv", parse_dates=True).values.tolist()
    # convert to datetime the 0th element of each row
    points = [[datetime.datetime.strptime(
        point[0], "%Y-%m-%d %H:%M"), point[1], point[2], point[3], point[4], point[5]] for point in points]

    print("Grouping data by day...")
    points_by_day = []
    # Group points by day
    for point in points:
        point.append(2)
        # convert timestamp to time2vec embedding
        if len(points_by_day) == 0:
            points_by_day.append([point])
        else:
            if points_by_day[-1][0][0].day == point[0].day:
                points_by_day[-1].append(point)
            else:
                points_by_day.append([point])


    print("Saving data...")
    # save data to pickle file
    with open(f"data/datasets/{symbol}_M{interval}.pkl", "wb") as f:
        pickle.dump(points_by_day, f)

    print(f"Data: {len(points_by_day)} days, {len(points)} points (datetime + OHLCV)")
    print(f"Done! Saved to data/datasets/{symbol}_M{interval}.pkl")
    print(points_by_day[0][0])


if __name__ == "__main__":
    make_dataset("EURUSD", 1)
