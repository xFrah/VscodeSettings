# here goes the code for the categorical transformer
import math
import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Time2Vec(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Time2Vec, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim // 2)
        self.periodic = nn.Linear(input_dim, output_dim // 2)

    def forward(self, x):  # x: (S, 1, E)
        # Output: (S, 1, E)
        return torch.cat((self.linear(x), torch.sin(self.periodic(x))), -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):  # x: (S, 1, E)
        S = x.size(0)
        pe = torch.zeros(S, self.d_model)
        position = torch.arange(0, S, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Output: (S, 1, E)
        return self.dropout(x + pe[: x.size(0), :])


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, noutput, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.time2vec = Time2Vec(ninp, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(nhid, noutput)

    def forward(self, src):  # src: (S, 1, E)
        src = self.time2vec(src)  # Output: (S, 1, E)
        src = self.pos_encoder(src)  # Output: (S, 1, E)
        output = self.transformer_encoder(src)  # Output: (S, 1, E)
        output = self.decoder(output)  # Output: (S, 1, C)
        return output.squeeze(1)  # Output: (S, C)
