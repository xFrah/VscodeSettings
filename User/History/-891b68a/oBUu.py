import torch
import torch.nn as nn
import torch.optim as optim
from build_vocab import Vocabularizer
from final_arch import TransformerModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def normalize_data_list(data_list):
    # normalize data_list
    max_deviation = max(abs(p - data_list[0]) for p in data_list)
    data_list = [(0.5 + (p - data_list[0]) / (2.0 * max_deviation))
                 for p in data_list]
    return data_list\



def normalize_slice(df_slice):
    # normalize the columns
    prices_close = normalize_data_list(df_slice["Close"].values.tolist())
    df_slice.loc[:, "Close"] = prices_close

    prices_volume = normalize_data_list(df_slice["Volume"].values.tolist())
    df_slice.loc[:, "Volume"] = prices_volume

    prices_open = normalize_data_list(df_slice["Open"].values.tolist())
    df_slice.loc[:, "Open"] = prices_open

    prices_high = normalize_data_list(df_slice["High"].values.tolist())
    df_slice.loc[:, "High"] = prices_high

    prices_low = normalize_data_list(df_slice["Low"].values.tolist())
    df_slice.loc[:, "Low"] = prices_low

    return df_slice


def get_slice_at_index(index, batch_size):
    # sliding window
    # make an explicit copy of the data
    df_slice = df.iloc[index: index + batch_size].copy()

    # select columns
    df_slice = df_slice[["Close", "Volume", "Open", "High", "Low"]]

    # normalize data
    df_slice = normalize_slice(df_slice)

    # tensor of shape (numberOfCandles, 5)
    df_slice = torch.tensor(df_slice.values, dtype=torch.float32)

    # get the next 3 prices (normalized) to calculate the slope
    label_index = index + batch_size
    label_points_normalized = normalize_data_list(
        df.iloc[label_index: label_index + 3]["Close"].values)

    # Create an array of x values corresponding to the indices of the y_values
    x_values = np.arange(len(label_points_normalized))
    unnormalized_slope, y_intercept = np.polyfit(
        x_values, label_points_normalized, 1)
    fit_line_slope = (unnormalized_slope + 1) / 2
    # tensor of shape (1) with the slope of the line
    label = torch.tensor(fit_line_slope, dtype=torch.float32)
    return label, df_slice, unnormalized_slope, y_intercept


def validate_training(df_val, input_size, train_losses, val_losses, loss_function, model, device):
    # generate random int between 0 and len(df_val) - 50
    val_idx = np.random.randint(0, len(df_val) - 50)
    label, df_slice, _, _, _ = get_slice_at_index(val_idx, input_size)

    # Move the data to the device
    # news = news.to(device)
    df_slice = df_slice.to(device)
    label = label.to(device).squeeze()

    # Forward pass
    outputs = model(df_slice).squeeze()

    # Compute loss
    loss = loss_function(outputs, label)

    val_losses.append(loss.item())


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(r"data\datasets\EURUSD_M15.csv")

    # We don't need these from the dataset
    df.pop("Date")
    df.pop("Previous Day Loss")
    df.pop("Previous Day Gain")
    df.pop("Fibonacci Retracement Level (14)")
    df.pop("Parabolic SAR Trend")

    df_columns = len(df.columns) - 1

    # df is training data, df_val is validation data
    df, df_val = df[: int(len(df) * 0.85)], df[int(len(df) * 0.85):]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocabularizer = Vocabularizer()

    model = TransformerModel(
        d_model=5,  # 5 Open High Low Close Volume
        nhead=5,
        d_hid=30,  # 30 hidden units
        nlayers=6,  # 6 encoder layers
    )

    model = model.to(device)

    # Set learning rate
    learning_rate = 0.01

    # Choose optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set loss function
    loss_function = nn.MSELoss()

    # Number of epochs
    epochs = 10

    # Size of an input (50 candles)
    input_size = 50

    # sliding window: 1 new datapoint per input, so we have len(df) - 3 inputs (3 because we need 3 points to calculate the slope)
    num_inputs = len(df) - 3

    torch.autograd.set_detect_anomaly(False)

    # import time
    train_losses = []
    val_losses = []
    i = 0

    for epoch in range(epochs):
        # shuffle range(num_inputs)
        with tqdm(range(num_inputs), desc=f"Epoch {epoch+1}/{epochs}", unit="input", position=0, leave=True) as pbar:
            shuffled = np.random.permutation(range(num_inputs - 50))
            for s_idx, _ in zip(shuffled, pbar):
                label, df_slice, gt_m, q = get_slice_at_index(
                    s_idx, input_size)

                # Move the data to the device
                # news = news.to(device)
                df_slice = df_slice.to(device)
                label = label.to(device).squeeze()

                # Forward pass
                outputs = model(df_slice).squeeze()

                # Compute loss
                loss = loss_function(outputs, label)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                if s_idx % 250 == 0:
                    i += 1
                    if i > 60:
                        for g in optimizer.param_groups:
                            g["lr"] = 1e-4
                    torch.save(model.state_dict(), "data/weights/model.pt")

                    # create a new figure
                    plt.clf()
                    # limit y range from 0 to 1
                    plt.ylim(0, 1)
                    # ! pred_list was removed
                    # plt.plot(pred_list, label="Actual", color="blue")
                    # plot vertical red line at 50
                    plt.axvline(x=50, color="red", linestyle="--")
                    # get scalar out of output
                    m = outputs.detach().cpu().numpy()
                    m = (m * 2) - 1

                    # plot line with slope m and y-intercept q that starts at x = 50
                    plt.plot([50, 53], [q, q + m * 3],
                             label="Predicted", color="orange")

                    # now plot real line with slope gt_m and y-intercept q that starts at x = 50
                    plt.plot([50, 53], [q, q + gt_m * 3],
                             label="Ground Truth", color="green")

                    plt.show(block=False)
                    plt.pause(0.001)

                    train_losses.append(loss.item())
                    # get the slope from train_losses[0] to train_losses[-1]
                    train_slope = (train_losses[-1] - train_losses[0]) / \
                        len(train_losses)
                    pbar.set_postfix(
                        {"Loss": loss.item(), "Train Slope": train_slope})

                    validate_training(
                        df_val, input_size, train_losses, val_losses, loss_function, model, device)
