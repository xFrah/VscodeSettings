import math
import torch
from torch.nn.functional import relu, softmax
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentivePooling, self).__init__()
        # Linear layer to compute attention scores
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        # inputs: output from Transformer encoder
        # Shape of inputs: [batch_size, seq_length, hidden_dim]

        attention_scores = self.attention(inputs)  # Apply linear layer
        # Shape of attention_scores: [batch_size, seq_length, 1]

        # Remove last dimension
        attention_scores = attention_scores.squeeze(-1)
        # Shape of attention_scores: [batch_size, seq_length]

        # Normalize scores with softmax
        attention_weights = softmax(attention_scores, dim=1)
        # Shape of attention_weights: [batch_size, seq_length]

        # Attention_weights.unsqueeze(-1) makes the shape: [batch_size, seq_length, 1]
        # This allows element-wise multiplication with inputs
        weighted_sum = (inputs * attention_weights.unsqueeze(-1)).sum(dim=1)  # Weighted sum of input vectors
        # Shape of weighted_sum: [batch_size, hidden_dim]
        # This can now be used as input to a linear layer for regression or classification

        return weighted_sum


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, d_hid, nlayers, dropout=0.25):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.att_pool = AttentivePooling(d_model)
        self.linear_mu = nn.Linear(d_model, 1)  # mean prediction
        self.linear_sigma = nn.Linear(d_model, 1)  # standard deviation prediction
        self.softplus = nn.Softplus()  # ensure std dev is > 0

    def forward(self, src_characteristics_vector, src_mask=None):
        src_characteristics_vector = src_characteristics_vector.unsqueeze(0)
        src_sentences = src_characteristics_vector * math.sqrt(self.d_model)
        src_sentences = self.pos_encoder(src_sentences)
        output = self.transformer_encoder(src_sentences, src_mask)
        pooled_output = self.att_pool(output)
        mu = self.linear_mu(pooled_output)
        sigma = self.softplus(self.linear_sigma(pooled_output))
        return mu.squeeze(), sigma.squeeze()



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
