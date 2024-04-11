import torch
import torch.nn as nn
import torch.optim as optim
from build_vocab import Vocabularizer
from final_arch import TransformerModel
from tqdm import tqdm


def make_news_batch(sentence_tensors):
    # output must be of shape [seq_len, batch_size]
    # seq_len is the length of the longest sentence in the batch
    # batch_size is the number of sentences in the batch

    # find the length of the longest sentence in the batch
    max_len = max([len(sentence) for sentence in sentence_tensors])

    # create a tensor of shape [seq_len, batch_size]
    batch = torch.zeros(max_len, len(sentence_tensors), dtype=torch.long)

    # fill the tensor with the sentences
    for i, sentence in enumerate(sentence_tensors):
        batch[: len(sentence), i] = sentence

    return batch


def get_slice_at_index(index, batch_size):
    # sliding window
    df_slice = df.iloc[index : index + batch_size].copy()  # make an explicit copy of the data
    # pop column from dataframe
    news = df_slice.pop("News").values
    start = df_slice.iloc[0]["Open"]
    prices = df_slice.pop("Close").values
    max_deviation = max(abs(p - start) for p in prices)

    prices = [(0.5 + (p - start) / (2.0 * max_deviation)) for p in prices]

    label_index = index + batch_size
    points = [(0.5 + (p - start) / (2.0 * max_deviation)) for p in df.iloc[label_index : label_index + 3]["Close"]]
    # Create an array of x values corresponding to the indices of the y_values
    x_values = np.arange(len(points))

    # Use numpy's polyfit function with degree 1 to get the slope and y-intercept of the best fit line
    unnormalized_slope, y_intercept = np.polyfit(x_values, points, 1)
    slope = (unnormalized_slope + 1) / 2

    # substitute old prices in dataframe
    df_slice.loc[:, "Close"] = prices  # using .loc[] to avoid SettingWithCopyWarning

    # now get version of dataframe with just column close
    df_slice = df_slice[["Close"]]

    # make tensor out of df_slice, each row is a vector, and each column is a dimension of this vector
    df_slice = torch.tensor(df_slice.values, dtype=torch.float32)
    label = torch.tensor(slope, dtype=torch.float32).reshape(-1, 1)
    prediction_slice = prices + [
        (0.5 + (p - start) / (2.0 * max_deviation)) for p in df.iloc[label_index : label_index + 3]["Close"].values
    ]  # make an explicit copy of the data

    return label, df_slice, prediction_slice, unnormalized_slope, y_intercept


def plot_training_progress(train_losses, val_losses, epoch, epochs):
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")

    data_mean = sum(train_losses[-15:]) / 15
    val_mean = sum(val_losses[-15:]) / 15
    # val_std = np.std(val_losses)
    plt.title(f"Epoch: {epoch}/{epochs}, Loss: {data_mean}, Val Loss: {val_mean}")
    # y_lower = val_mean - 2 * val_std
    # y_upper = val_mean + 2 * val_std
    # limit y axis to 0-1
    # plt.ylim(y_lower, y_upper)
    plt.yscale("log")
    plt.show(block=False)
    plt.pause(0.001)


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
    df.pop("Volume")
    df.pop("Previous Day Loss")
    df.pop("Previous Day Gain")
    df.pop("Fibonacci Retracement Level (14)")
    df.pop("Parabolic SAR Trend")

    df_columns = len(df.columns) - 1

    # df is training data, df_val is validation data
    df, df_val = df[: int(len(df) * 0.85)], df[int(len(df) * 0.85) :]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocabularizer = Vocabularizer()
    # sentences = pd.read_csv(r"data\news\forexlive.csv")["Article"].astype(str).tolist()
    # src_sentences = make_batch([vocabularizer(sentence) for sentence in sentences[:batch_size]])
    # src_sentences = torch.randint(0, 100, (20, 20))
    # src_characteristics_vector = torch.randn(batch_size, batch_size)

    model = TransformerModel(
        d_model=1,
        nhead=1,
        d_hid=20,
        nlayers=3,
    )

    # model = nn.DataParallel(model)
    model = model.to(device)

    # output = model(src_sentences, src_characteristics_vector)
    # print(output.shape)  # torch.Size([5, 15])
    # print(output)

    # Set learning rate
    learning_rate = 1e-5
    # Choose optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set loss function
    loss_function = nn.MSELoss()

    # Number of epochs
    epochs = 10

    # Size of an input (50 bars)
    input_size = 50

    # sliding window: 1 new datapoint per input, so we have len(df) - 2 inputs
    num_inputs = len(df) - 2

    torch.autograd.set_detect_anomaly(False)
    import matplotlib.pyplot as plt
    import numpy as np

    # import time
    train_losses = []
    val_losses = []
    i = 0

    for epoch in range(epochs):
        # shuffle range(num_inputs)
        with tqdm(range(num_inputs), desc=f"Epoch {epoch+1}/{epochs}", unit="input", position=0, leave=True) as pbar:
            shuffled = np.random.permutation(range(num_inputs - 50))
            for s_idx, _ in zip(shuffled, pbar):
                label, df_slice, pred_list, gt_m, q = get_slice_at_index(s_idx, input_size)

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
                    if i > 60:
                        for g in optimizer.param_groups:
                            g["lr"] = 0.01
                    torch.save(model.state_dict(), "data/weights/model.pt")

                    # create a new figure
                    plt.clf()
                    # limit y range from 0 to 1
                    plt.ylim(0, 1)
                    plt.plot(pred_list, label="Actual", color="blue")
                    # plot vertical red line at 50
                    plt.axvline(x=50, color="red", linestyle="--")
                    # get scalar out of output
                    m = outputs.detach().cpu().numpy()
                    m = (m * 2) - 1

                    # plot line with slope m and y-intercept q that starts at x = 50
                    plt.plot([50, 53], [q, q + m * 3], label="Predicted", color="orange")

                    # now plot real line with slope gt_m and y-intercept q that starts at x = 50
                    plt.plot([50, 53], [q, q + gt_m * 3], label="Ground Truth", color="green")

                    plt.show(block=False)
                    plt.pause(0.001)

                    train_losses.append(loss.item())
                    pbar.set_postfix({"loss": loss.item()})

                    validate_training(df_val, input_size, train_losses, val_losses, loss_function, model, device)

                    # plot_training_progress(train_losses, val_losses, epoch, epochs)
