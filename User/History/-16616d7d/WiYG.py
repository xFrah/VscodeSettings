# here goes the code for the categorical transformer
import math
import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Time2Vec(nn.Module):
    def __init__(self, input_size, output_size):
        super(Time2Vec, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        time_linear = self.linear(x)
        time_sin = torch.sin(time_linear)
        return torch.cat([x, time_sin], -1)


class MyTransformerEncoder(nn.Module):
    def __init__(self, input_size, nhead, num_layers):
        super(MyTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class ClassificationLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationLayer, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class StockPredictionModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(StockPredictionModel, self).__init__()
        # Adds a learnable temporal dimension to the input
        self.time2vec = Time2Vec(d_model, 1)
        # Transforms the input using self-attention mechanisms
        self.transformer = TransformerEncoder(d_model + 1, nhead, num_layers)
        self.classification = ClassificationLayer(d_model + 1, num_classes)  # Classifies the transformed input

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        # Output shape: (batch_size, sequence_length, d_model + 1)
        x = self.time2vec(x)
        # Output shape: (batch_size, sequence_length, d_model + 1)
        x = self.transformer(x)

        # Take the mean of the transformer's output across the sequence length dimension
        x = x.mean(dim=1)  # Output shape: (batch_size, d_model + 1)

        x = self.classification(x)  # Output shape: (batch_size, num_classes)
        return x  # Output shape: (batch_size, num_classes)
