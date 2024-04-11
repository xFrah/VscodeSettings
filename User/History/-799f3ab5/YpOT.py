import numpy as np
import pandas as pd


def make_dataset(csvName: str) -> None:
    """
    This function creates a dataset from a csv file.
    """
    # Read csv file
    data = pd.read_csv(csvName, index_col=0)
    sequences = []
    # Create sequences
    # Create sequences of 50 days each, using a sliding window of 1 day
    # sequences are tuples (x, y) where x is the sequence of 50 days and y is the price of the 51st day
    for i in range(len(data) - 50):
        x = data.iloc[i : i + 50]
        # turn x into an array
        x = x.to_numpy()
        change = data.iloc[i + 50]["Previous Day Change"]
        y = 0
        if change > 0.0001:
            y = 1
        elif change < -0.0001:
            y = -1
        sequences.append([x, y])
    # Turn sequences into a numpy array
    sequences = np.array(sequences, dtype=object)
    # Save sequences to a file
    np.save("prices/dataset.npy", sequences)


if __name__ == "__main__":
    make_dataset("prices/pricedata/EURUSD_M15_ANALYZED.csv")
    ds = np.load("prices/dataset.npy", allow_pickle=True)
    print(ds.shape)
    pass
