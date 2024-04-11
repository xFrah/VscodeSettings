import datetime
import pandas as pd
import os
import pickle
import numpy as np


def normalize_prices(points: list) -> None:
    # for each point, normalize the OHLCV data (index 1-5)
    for i in range(1, 6):
        data_list = [point[i] for point in points]
        data_list, _, _ = normalize_data_list(data_list)
        for j in range(len(points)):
            points[j][i] = data_list[j]


def normalize_data_list(data_list: list[float], start=None, maxdev=None) -> tuple[list, float, float]:
    # normalize data_list
    if start is None:
        start = data_list[0]
    if maxdev is None:
        maxdev = max(abs(p - start) for p in data_list) * 2.0
    return [(0.5 + (p - start) / (maxdev + 0.000001)) for p in data_list], start, maxdev


def make_dataset(symbol: str, interval: int) -> None:
    print(f"Making dataset for {symbol} with interval {interval}...")
    # Read data from csv file
    points = pd.read_csv(f"data/prices/{symbol}_M{interval}.csv", parse_dates=True).values.tolist()
    # convert to datetime the 0th element of each row
    points = [
        [datetime.datetime.strptime(point[0], "%Y-%m-%d %H:%M"), point[1], point[2], point[3], point[4], point[5]] for point in points
    ]

    print("Grouping data by day...")
    points_by_day = []
    # Group points by day
    for point in points:
        point.append()
        # convert timestamp to time2vec embedding
        if len(points_by_day) == 0:
            points_by_day.append([point])
        else:
            if points_by_day[-1][0][0].day == point[0].day:
                points_by_day[-1].append(point)
            else:
                points_by_day.append([point])

    # turn datetime into (hour*60+minute)/1440
    for day in points_by_day:
        for point in day:
            point[0] = (point[0].hour * 60 + point[0].minute) / 1440
        normalize_prices(day)

    print("Saving data...")
    # save data to pickle file
    with open(f"data/datasets/{symbol}_M{interval}.pkl", "wb") as f:
        pickle.dump(points_by_day, f)

    print(f"Data: {len(points_by_day)} days, {len(points)} points (datetime + OHLCV)")
    print(f"Done! Saved to data/datasets/{symbol}_M{interval}.pkl")
    print(points_by_day[0][0:5])


if __name__ == "__main__":
    make_dataset("EURUSD", 1)
