import torch
import torch.nn as nn
import torch.optim as optim
from build_vocab import Vocabularizer
from final_arch import TransformerModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def normalize_data_list(data_list, start=None, maxdev=None) -> tuple[list, float, float]:
    # normalize data_list
    if start is None:
        start = data_list[0]
    if maxdev is None:
        max_deviation = max(abs(p - start) for p in data_list)
    return [(0.5 + (p - start) / (2.0 * max_deviation + 0.000001)) for p in data_list], start, maxdev


def normalize_slice(df_slice):
    # normalize the columns
    prices_close = normalize_data_list(df_slice["Close"].values)
    df_slice.loc[:, "Close"] = prices_close

    prices_open = normalize_data_list(df_slice["Open"].values)
    df_slice.loc[:, "Open"] = prices_open

    prices_volume = normalize_data_list(df_slice["Volume"].values)
    df_slice.loc[:, "Volume"] = prices_volume

    prices_high = normalize_data_list(df_slice["High"].values)
    df_slice.loc[:, "High"] = prices_high

    prices_low = normalize_data_list(df_slice["Low"].values)
    df_slice.loc[:, "Low"] = prices_low

    rsi = normalize_data_list(
        df_slice["Relative Strength Index"].values.tolist())
    df_slice.loc[:, "Relative Strength Index"] = rsi

    simple_moving_average_20 = normalize_data_list(
        df_slice["Simple Moving Average (20)"].values.tolist())
    df_slice.loc[:, "Simple Moving Average (20)"] = simple_moving_average_20

    standard_deviation_20 = normalize_data_list(
        df_slice["Standard Deviation (20)"].values.tolist())
    df_slice.loc[:, "Standard Deviation (20)"] = standard_deviation_20

    bollinger_band_20_2_upper = normalize_data_list(
        df_slice["Bollinger Band (20,2) Upper"].values.tolist())
    df_slice.loc[:, "Bollinger Band (20,2) Upper"] = bollinger_band_20_2_upper

    bollinger_band_20_2_lower = normalize_data_list(
        df_slice["Bollinger Band (20,2) Lower"].values.tolist())
    df_slice.loc[:, "Bollinger Band (20,2) Lower"] = bollinger_band_20_2_lower

    exponential_moving_average_12 = normalize_data_list(
        df_slice["Exponential Moving Average (12)"].values.tolist())
    df_slice.loc[:,
                 "Exponential Moving Average (12)"] = exponential_moving_average_12

    exponential_moving_average_26 = normalize_data_list(
        df_slice["Exponential Moving Average (26)"].values.tolist())
    df_slice.loc[:,
                 "Exponential Moving Average (26)"] = exponential_moving_average_26

    macd_line = normalize_data_list(
        df_slice["MACD Line"].values.tolist())
    df_slice.loc[:, "MACD Line"] = macd_line

    macd_signal = normalize_data_list(
        df_slice["MACD Signal"].values.tolist())
    df_slice.loc[:, "MACD Signal"] = macd_signal

    previous_day_change = normalize_data_list(
        df_slice["Previous Day Change"].values.tolist())
    df_slice.loc[:, "Previous Day Change"] = previous_day_change

    average_gain_ewm_13_14 = normalize_data_list(
        df_slice["Average Gain EWM (13,14)"].values.tolist())
    df_slice.loc[:, "Average Gain EWM (13,14)"] = average_gain_ewm_13_14

    average_loss_ewm_13_14 = normalize_data_list(
        df_slice["Average Loss EWM (13,14)"].values.tolist())
    df_slice.loc[:, "Average Loss EWM (13,14)"] = average_loss_ewm_13_14

    relative_strength = normalize_data_list(
        df_slice["Relative Strength"].values.tolist())
    df_slice.loc[:, "Relative Strength"] = relative_strength

    simple_moving_average_34 = normalize_data_list(
        df_slice["Simple Moving Average (34)"].values.tolist())
    df_slice.loc[:, "Simple Moving Average (34)"] = simple_moving_average_34

    awesome_oscillator = normalize_data_list(
        df_slice["Awesome Oscillator"].values.tolist())
    df_slice.loc[:, "Awesome Oscillator"] = awesome_oscillator

    lowest_low_14 = normalize_data_list(
        df_slice["Lowest Low (14)"].values.tolist())
    df_slice.loc[:, "Lowest Low (14)"] = lowest_low_14

    highest_high_14 = normalize_data_list(
        df_slice["Highest High (14)"].values.tolist())
    df_slice.loc[:, "Highest High (14)"] = highest_high_14

    stochastic_fast_signal_k = normalize_data_list(
        df_slice["Stochastic Fast Signal (%K)"].values.tolist())
    df_slice.loc[:, "Stochastic Fast Signal (%K)"] = stochastic_fast_signal_k

    stochastic_slow_signal_d = normalize_data_list(
        df_slice["Stochastic Slow Signal (%D)"].values.tolist())
    df_slice.loc[:, "Stochastic Slow Signal (%D)"] = stochastic_slow_signal_d

    accumulation_distribution = normalize_data_list(
        df_slice["Accumulation Distribution"].values.tolist())
    df_slice.loc[:, "Accumulation Distribution"] = accumulation_distribution

    psar = normalize_data_list(
        df_slice["PSAR"].values.tolist())
    df_slice.loc[:, "PSAR"] = psar

    parabolic_sar_extreme_point = normalize_data_list(
        df_slice["Parabolic SAR Extreme Point"].values.tolist())
    df_slice.loc[:, "Parabolic SAR Extreme Point"] = parabolic_sar_extreme_point

    parabolic_sar_trend = normalize_data_list(
        df_slice["Parabolic SAR Trend"].values.tolist())
    df_slice.loc[:, "Parabolic SAR Trend"] = parabolic_sar_trend

    return df_slice


def get_slice_at_index(index, batch_size):
    # sliding window
    # make an explicit copy of the data
    df_slice = df.iloc[index: index + batch_size].copy()

    # select columns
    df_slice = df_slice[["Close", "Volume", "Open",
                         "High", "Low", "Relative Strength Index", "Simple Moving Average (20)", "Standard Deviation (20)",
                         "Bollinger Band (20,2) Upper", "Bollinger Band (20,2) Lower", "Exponential Moving Average (12)",
                         "Exponential Moving Average (26)", "MACD Line", "MACD Signal",
                         "Previous Day Change", "Average Gain EWM (13,14)", "Average Loss EWM (13,14)", "Relative Strength", "Simple Moving Average (34)",
                         "Awesome Oscillator", "Lowest Low (14)", "Highest High (14)", "Stochastic Fast Signal (%K)",
                         "Stochastic Slow Signal (%D)", "Accumulation Distribution", "PSAR", "Parabolic SAR Extreme Point", "Parabolic SAR Trend"]]

    # normalize data
    df_slice, start, maxdev = normalize_slice(df_slice)

    # get the next 3 prices (normalized) to calculate the slope
    label_index = index + batch_size
    label_points_normalized = normalize_data_list(
        df.iloc[label_index: label_index + 3]["Close"].values, start=start, maxdev=maxdev)

    prediction_slice = df_slice["Close"].values + label_points_normalized

    # Create an array of x values corresponding to the indices of the y_values
    x_values = np.arange(len(label_points_normalized))
    unnormalized_slope, y_intercept = np.polyfit(
        x_values, label_points_normalized, 1)
    fit_line_slope = (unnormalized_slope + 1) / 2
    # tensor of shape (1) with the slope of the line
    label = torch.tensor(fit_line_slope, dtype=torch.float32)
    # tensor of shape (numberOfCandles, 5)
    df_slice = torch.tensor(df_slice.values, dtype=torch.float32)
    return label, df_slice, unnormalized_slope, y_intercept, prediction_slice


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
    df, df_val = df[: int(len(df) * 0.85)], df[int(len(df) * 0.85):]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocabularizer = Vocabularizer()

    model = TransformerModel(
        d_model=28,
        nhead=4,
        d_hid=25,  # 30 hidden units
        nlayers=3,  # 6 encoder layers
        dropout=0.65,
    )

    model = model.to(device)

    # Set learning rate
    learning_rate = 0.001

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

    # create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    for epoch in range(epochs):
        # shuffle range(num_inputs)
        with tqdm(range(num_inputs), desc=f"Epoch {epoch+1}/{epochs}", unit="input", position=0, leave=True) as pbar:
            shuffled = np.random.permutation(range(num_inputs - 50))
            for s_idx, _ in zip(shuffled, pbar):
                label, df_slice, gt_m, q, pred_list = get_slice_at_index(
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

                if s_idx % 50 == 0:
                    i += 1
                    if i > 300:
                        for g in optimizer.param_groups:
                            g["lr"] = 1e-3
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
                    ax2.plot([50, 53], [q, q + m * 3],
                             label="Predicted", color="orange")

                    # # now plot real line with slope gt_m and y-intercept q that starts at x = 50
                    ax2.plot([50, 53], [q, q + gt_m * 3],
                             label="Ground Truth", color="green")

                    # now show
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
