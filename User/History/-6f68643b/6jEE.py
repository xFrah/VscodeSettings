# here goes the code for the categorical transformer
import torch.nn as nn
import torch


class Time2Vec2(nn.Module):
    def __init__(self):
        super(Time2Vec2, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x_time):
        time_linear = self.linear(x_time)
        time_sin = torch.sin(time_linear)
        return torch.cat([x_time, time_sin], -1)


class TransformerEncoder2(nn.Module):
    def __init__(self, input_size, nhead, num_layers):
        super(TransformerEncoder2, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class ClassificationLayer2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationLayer2, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class StockPredictionModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(StockPredictionModel, self).__init__()
        # Adds a learnable temporal dimension to the input
        self.time2vec = Time2Vec2()
        # Transforms the input using self-attention mechanisms
        self.transformer = TransformerEncoder2(d_model + 1, nhead, num_layers)
        self.classification = ClassificationLayer2(d_model + 1, num_classes)  # Classifies the transformed input

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        
        # Extract the time feature and apply Time2Vec to it
        x_time = x[..., :1]  # shape: (batch_size, sequence_length, 1)
        x_time = self.time2vec(x_time)  # shape: (batch_size, sequence_length, 2)
        
        # Concatenate the time feature with the rest of the features
        x = torch.cat([x_time, x[..., 1:]], dim=-1)  # shape: (batch_size, sequence_length, d_model + 1)
        
        # Apply the transformer
        x = self.transformer(x)  # shape: (batch_size, sequence_length, d_model + 1)

        # Take the mean of the transformer's output across the sequence length dimension
        x = x.mean(dim=1)  # shape: (batch_size, d_model + 1)

        x = self.classification(x)  # shape: (batch_size, num_classes)
        return x  # shape: (batch_size, num_classes)