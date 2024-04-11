import datetime
import pandas as pd
import os
import pickle
import numpy as np


def time2vec_embedding(timestamp):
    # Convert timestamp to minutes since midnight
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute

    # Convert to radians to create Time2Vec embeddings
    time_embedding = [
        np.sin(2 * np.pi * minutes_since_midnight / (24 * 60)),
        np.cos(2 * np.pi * minutes_since_midnight / (24 * 60))
    ]

    return time_embedding


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
        time2vec_res = time2vec_embedding(point[0])
        point.append(time2vec_res)
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

    print(f"Data: {len(points_by_day)} days, {len(points)} points")
    print(f"Done! Saved to data/datasets/{symbol}_M{interval}.pkl")


if __name__ == "__main__":
    make_dataset("EURUSD", 1)
