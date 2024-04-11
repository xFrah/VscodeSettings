import torch
import torch.nn as nn
import torch.optim as optim
from build_vocab import Vocabularizer
from final_arch import TransformerModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

columns_used = ["Close", "Volume", "Open", "High", "Low", "Standard Deviation (20)"]


def normalize_data_list(data_list, start=None, maxdev=None) -> tuple[list, float, float]:
    # normalize data_list
    if start is None:
        start = data_list[0]
    if maxdev is None:
        maxdev = max(abs(p - start) for p in data_list) * 2.0
    return [(0.5 + (p - start) / (maxdev + 0.000001)) for p in data_list], start, maxdev


def normalize_slice(df_slice):
    # normalize the columns
    for col in columns_used:
        prices_close, _, _ = normalize_data_list(df_slice[col].values)
        df_slice.loc[:, col] = prices_close


def get_slice_at_index(index, batch_size):
    # sliding window
    # make an explicit copy of the data
    normalized = df.iloc[index : index + batch_size].copy()
    unnormalized = df.iloc[index : index + batch_size].copy()

    # select columns
    normalized = normalized[columns_used]

    # normalize data
    normalize_slice(normalized)

    norm, start, maxdev = normalize_data_list(unnormalized["Close"].values)

    # get the next 3 prices (normalized) to calculate the slope
    label_index = index + batch_size
    label_points_normalized, _, _ = normalize_data_list(df.iloc[label_index : label_index + 3]["Close"].values, start=start, maxdev=maxdev)

    prediction_slice = norm + label_points_normalized

    # Create an array of x values corresponding to the indices of the y_values
    x_values = np.arange(len(label_points_normalized))
    unnormalized_slope, y_intercept = np.polyfit(x_values, label_points_normalized, 1)
    fit_line_slope = (unnormalized_slope + 1) / 2
    # tensor of shape (1) with the slope of the line
    label = torch.tensor(fit_line_slope, dtype=torch.float32)
    # tensor of shape (numberOfCandles, 5)
    normalized = torch.tensor(normalized.values, dtype=torch.float32)
    return label, normalized, unnormalized_slope, y_intercept, prediction_slice


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

    df_columns = len(df.columns) - 1

    # df is training data, df_val is validation data
    df, df_val = df[: int(len(df) * 0.85)], df[int(len(df) * 0.85) :]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocabularizer = Vocabularizer()

    model = TransformerModel(
        d_model=6,
        nhead=3,
        d_hid=30,  # 30 hidden units
        nlayers=3,  # 3 encoder layers
        dropout=0,
    )

    model = model.to(device)

    # Set learning rate
    learning_rate = 1e-5

    # Choose optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set loss function
    loss_function = nn.CrossEntropyLoss()

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

    # create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    for epoch in range(epochs):
        # shuffle range(num_inputs)
        with tqdm(range(num_inputs), desc=f"Epoch {epoch+1}/{epochs}", unit="input", position=0, leave=True) as pbar:
            shuffled = np.random.permutation(range(num_inputs - 50))
            for s_idx, _ in zip(shuffled, pbar):
                label, df_slice, gt_m, q, pred_list = get_slice_at_index(s_idx, input_size)

                # Move the data to the device
                # news = news.to(device)
                df_slice = df_slice.to(device)
                label = label.to(device).squeeze()

                # Forward pass
                outputs = model(df_slice)  # No need to squeeze since outputs are now a batch of distributions

                # Compute loss
                loss = loss_function(outputs, label.long())  # label needs to be a LongTensor for CrossEntropyLoss

                # Zero the gradients
                optimizer.zero_grad()

                # Perform backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                if s_idx % 100 == 0:
                    i += 1
                    torch.save(model.state_dict(), "data/weights/model.pt")

                    # clear figures
                    ax1.clear()
                    ax2.clear()

                    # plot the train losses
                    ax1.plot(train_losses, label="Train Loss", color="blue")
                    ax1.set_yscale("log")
                    ax1.set_title("Train Loss")
                    # # create a new figure
                    # plt.clf()
                    # # limit y range from 0 to 1
                    # plt.ylim(0, 1)
                    # set ax2 limit y range from 0 to 1
                    ax2.set_ylim(0, 1)
                    # # ! pred_list was removed
                    ax2.plot(pred_list, label="Actual", color="blue")
                    # # plot vertical red line at 50
                    ax2.axvline(x=50, color="red", linestyle="--")
                    # # get scalar out of output
                    m = outputs.detach().cpu().numpy()
                    m = (m * 2) - 1

                    # # plot line with slope m and y-intercept q that starts at x = 50
                    ax2.plot([50, 53], [q, q + m * 3], label="Predicted", color="orange")

                    # # now plot real line with slope gt_m and y-intercept q that starts at x = 50
                    ax2.plot([50, 53], [q, q + gt_m * 3], label="Ground Truth", color="green")

                    # now show
                    plt.show(block=False)
                    plt.pause(0.001)

                    train_losses.append(loss.item())
                    # get the slope from train_losses[0] to train_losses[-1]
                    train_slope = (train_losses[-1] - train_losses[0]) / len(train_losses)
                    pbar.set_postfix({"Loss": loss.item(), "Train Slope": train_slope})

                    validate_training(df_val, input_size, train_losses, val_losses, loss_function, model, device)
