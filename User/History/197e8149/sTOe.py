import math
import torch
from torch.nn.functional import relu
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        nhead2: int,
        d_hid: int,
        nlayers: int,
        characteristics_dimensionality: int,
        neurons: list[int],
        final_neurons: list[int],
        dropout: float = 0.5,
        conv_kernel_size=2,
        final_conv_kernel_size=2,
    ):
        super().__init__()
        self.characteristics_dimensionality = characteristics_dimensionality
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers_1 = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder_1 = TransformerEncoder(
            encoder_layers_1, nlayers)
        encoder_layers_2 = TransformerEncoderLayer(
            neurons[-1] + characteristics_dimensionality, nhead2, d_hid, dropout)
        self.transformer_encoder_2 = TransformerEncoder(
            encoder_layers_2, nlayers)
        # d_model is the dimension of the embedding
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.last_conv_length = neurons[-1]

        self.convs = nn.ModuleList()  # list to hold convolutional layers
        in_channels = d_model  # the number of input channels for the first conv layer is d_model
        for out_channels in neurons:
            self.convs.append(nn.Conv1d(in_channels, out_channels,
                              kernel_size=conv_kernel_size))  # creating a conv layer
            # the number of input channels for the next layer is the number of output channels of the current layer
            in_channels = out_channels

        assert (
            neurons[-1] == final_neurons[0]
        ), "The output of the last conv layer should have the same number of neurons as the first element of the final_conv_channels list"

        self.final_convs = nn.ModuleList()
        final_in_channels = neurons[-1] + characteristics_dimensionality
        for out_channels in final_neurons:
            self.final_convs.append(
                nn.Conv1d(final_in_channels, out_channels,
                          kernel_size=final_conv_kernel_size if out_channels != 1 else 1)
            )
            final_in_channels = out_channels

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for conv in self.convs:
            conv.bias.data.zero_()
            conv.weight.data.uniform_(-initrange, initrange)
        for conv in self.final_convs:
            conv.bias.data.zero_()
            conv.weight.data.uniform_(-initrange, initrange)

    def forward(self, src_sentences: Tensor, src_characteristics_vector: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src_sentences: Tensor, shape ``[seq_len, batch_size]`` --> Input shape
            src_characteristics_vector: Tensor, shape ``[batch_size, characteristics_dim]`` --> Characteristics vector shape
            src_mask: Tensor, shape ``[seq_len, seq_len]`` --> Mask shape

        Returns:
            output Tensor of shape ``[batch_size, neurons[-1]]`` where neurons[-1] is the number of neurons in the final conv layer.
        """

        assert (
            self.characteristics_dimensionality == src_characteristics_vector.shape[1]
        ), f"The characteristics vector should have the same number of dimensions as the characteristics_dimensionality parameter({src_characteristics_vector.shape[1]} != {self.characteristics_dimensionality})"

        a = self.embedding(src_sentences)
        # Shape after embedding: [seq_len, batch_size, d_model]
        src_sentences = a * math.sqrt(self.d_model)
        # Positional encoding doesn't change the shape: [seq_len, batch_size, d_model]
        src_sentences = self.pos_encoder(src_sentences)
        # Transformer output: [seq_len, batch_size, d_model]
        output = self.transformer_encoder_1(src_sentences, src_mask)

        # for the first sentence, make dot product between every pair of words and print it as a list

        # sentence = output[:, 0, :]
        # for j in range(sentence.shape[0]):
        #    line = ""
        #    for i in range(sentence.shape[0]):
        #        dot = torch.dot(sentence[i], sentence[j])
        #        try:
        #            line += str(int(dot)) + " "
        #        except:
        #            line += "nan "
        #    print(line)

        # Change the order of the dimensions for the Conv1d layer
        # Output shape: [batch_size, d_model, seq_len]
        output = output.permute(1, 2, 0)
        for conv in self.convs:
            # After each Conv1d layer: [batch_size, neurons[i], seq_len],
            output = relu(conv(output))
            # where neurons[i] is the number of neurons in the i-th conv layer

        # Max pooling over the sequence length dimension
        # Output shape: [batch_size, neurons[-1]]
        output, _ = output.max(dim=-1)

        # [batch_size, characteristics_dim + neurons_dim]
        combined_embeddings = torch.cat(
            [src_characteristics_vector, output], dim=-1)

        assert (
            combined_embeddings.shape[1] == self.last_conv_length +
            src_characteristics_vector.shape[1]
        ), "The output of the last conv layer should have the same number of neurons as the last element of the neurons list"
        assert (
            combined_embeddings.shape[0] == src_sentences.shape[1]
        ), "The batch size of the output should be equal to the batch size of the input"

        # let's use the output of these steps as the input to a new transformer encoder
        # input shape [seq_len = previous_batch_size, 1, characteristics_dim + neurons_dim]
        output_2 = self.transformer_encoder_2(
            combined_embeddings, src_mask)  # output shape [seq_len, d_model]

        # reshape for the Conv1d layer
        # adding a dummy batch dimension. output shape: [1, seq_len, d_model]
        output_2 = output_2.unsqueeze(0)
        # output shape [1, d_model, seq_len]
        output_3 = output_2.permute(0, 2, 1)

        # final convolutional layers
        for conv in self.final_convs:
            output_3 = relu(conv(output_3))  # After each Conv1d layer

        # adaptive average pooling to reduce the final dimension
        # output shape [1, final_conv_channels[-1], 1]
        price_prediction = nn.AdaptiveAvgPool1d(1)(output_3)

        # squeeze the unnecessary dimensions
        # output shape [final_conv_channels[-1]]
        price_prediction = price_prediction.squeeze()

        return price_prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
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
