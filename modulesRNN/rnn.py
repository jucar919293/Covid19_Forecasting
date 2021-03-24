from abc import ABC
from typing import Any

import torch
import torch.nn as nn
from .weight_init import weight_init
import random
from torch.nn.utils.rnn import pad_sequence
"""
    Starting some variables
"""

device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

"""
Classes used in the architectures
"""


# noinspection PyAbstractClass
class InputAttention(nn.Module):

    def __init__(
            self,
            size_hidden: int = 128,
            n_features: int = 18,  # Number of features
            size_seq: int = 40,
    ) -> None:
        super(InputAttention, self).__init__()

        self.size_hidden = size_hidden
        self.n_features = n_features
        self.size_seq = size_seq

        self.linearHidden = nn.Linear(
            in_features=self.size_hidden,
            out_features=self.size_seq,
            bias=True
        )
        self.linearSignals = nn.Linear(
            in_features=self.size_seq,
            out_features=self.size_seq,
            bias=True
        )
        self.linearOut = nn.Linear(
            in_features=self.size_seq,
            out_features=1,
            bias=True
        )
        self.soft = nn.Softmax(1)

        # init weights
        weight_init(self.linearHidden)
        weight_init(self.linearSignals)
        weight_init(self.linearOut)

    def forward(self, signalK, h_t_previous):
        linear_hidens = self.linearHidden(h_t_previous)
        linear_signal = self.linearSignals(signalK)
        sum_linears = linear_hidens.add(linear_signal)
        tan_sum = torch.tanh(sum_linears)
        e_values_t = self.linearOut(tan_sum)  # eValues = (V(tanh(H*W1 +b1+x*W2 +b2)+b3)
        att_values_t = self.soft(e_values_t).squeeze(-1)
        return e_values_t, att_values_t


# noinspection PyAbstractClass
class EncoderAttention(nn.Module):
    def __init__(
            self,
            dim_seq_in: int = 5,
            rnn_out: int = 128,
            n_layers: int = 1,
            size_seq: int = 36,
            bidirectional: bool = False,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of features)
        param rnn_out: output dimension for rnn
        """
        super(EncoderAttention, self).__init__()
        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.bidirectional = bidirectional
        self.size_seq = size_seq

        # Creating Attention
        self.attention = InputAttention(size_hidden=self.rnn_out,
                                        n_features=self.dim_seq_in,
                                        size_seq=self.size_seq).to(device).type(dtype)

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            num_layers=n_layers,
            batch_first=True,
        )

        # init weights
        weight_init(self.rnn)

    def forward(self, seqs, mask=None):
        """
        param mask: binary tensor with same shape as seqs
        """

        hidden_total = []
        seqs_update = seqs.clone()
        h_t_previous = torch.zeros((seqs.shape[0], self.rnn_out)).unsqueeze(1).to(device).type(dtype)
        signal_k = seqs.transpose(1, 2)
        for t in range(seqs.shape[1]):
            h_t_previous_signals = h_t_previous.repeat(1, seqs.shape[-1], 1)
            # Obtaining attention values for time t
            e_values_t, att_values_t = self.attention(signal_k, h_t_previous_signals)
            seqs_update[:, t, :] = att_values_t * seqs[:, t, :]
            # Updating values for next step t+1
            h_t = self.rnn(seqs_update.clone())[0]
            hidden_total.append(h_t[:, t, :])
            h_t_previous = h_t[:, t, :].clone().unsqueeze(1)
        hidden_total = torch.stack(hidden_total, 1).to(device).type(dtype)
        # Pass through first rnn
        latent_seqs = hidden_total * mask  # keep hidden states that correspond to non-zero in seqs
        latent_seqs = latent_seqs.sum(1)  # NOTE: change when doing attention
        # pdb.set_trace()
        # pdb.set_trace()
        return latent_seqs


# noinspection PyAbstractClass
class Encoder(nn.Module):
    def __init__(
            self,
            dim_seq_in: int = 5,
            rnn_out: int = 128,
            n_layers: int = 1,
            bidirectional: bool = False,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of features)
        param dim_out: Dimensionality of output vector
        param rnn_out: output dimension for rnn
        """
        super(Encoder, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            num_layers=n_layers,
            batch_first=True,
        )
        # init weights
        weight_init(self.rnn)

    def forward(self, seqs, mask):
        """
        param mask: binary tensor with same shape as seqs
        """
        # pass through first rnn

        latent_seqs = self.rnn(seqs)[0]  # index 0 obtains all hidden states
        latent_seqs = latent_seqs * mask  # keep hidden states that correspond to non-zero in seqs
        latent_seqs = latent_seqs.sum(1)  # NOTE: change when doing attention
        # pdb.set_trace()
        return latent_seqs


# noinspection PyAbstractClass
class Decoder(nn.Module):
    def __init__(
            self,
            dim_seq_in: int = 5,
            rnn_out: int = 128,
            n_layers: int = 1,
            bidirectional: bool = False,
            dim_out: int = 1,
            dropout: float = 0.2,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (hiden state size)
        param rnn_out: output dimension for rnn
        """
        super(Decoder, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional
        self.dropout = dropout

        hidden_size = self.rnn_out // 2 if self.bidirectional else self.rnn_out

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.out_layer = [
            nn.Linear(
                in_features=hidden_size,
                out_features=int(hidden_size / 2)
            ),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=int(hidden_size / 2),
                out_features=int(hidden_size / 4)
            ),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(num_features=2*self.rnn_out,affine=True),
            # nn.Linear(
            #     in_features=2*self.rnn_out, out_features=2*self.rnn_out
            # ),
            # nn.Sigmoid(),
            nn.Linear(
                in_features=int(hidden_size / 4), out_features=int(hidden_size / 8)
            ),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=int(hidden_size / 8), out_features=self.dim_out
            ),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)
        # init weights
        weight_init(self.rnn)
        weight_init(self.out_layer)

    def forward(self, hidden, k_wk_ahead, ys_batch=None, teacher_forcing_ratio=0.5):
        """
        Teacher forcing

        param k_wk_ahead: how many weeks ahead to forecast
        """
        inputs = torch.zeros((hidden.shape[0], 1, 1), dtype=dtype).to(device)
        # note that hidden should be of (num_layers * num_directions, batch, hidden_size)
        hidden = hidden.unsqueeze(0)  # adding one dimension corresponding to num_layers * num_directions
        outputs = []

        for k in range(k_wk_ahead):
            # pass through first rnn
            latent_seqs = self.rnn(inputs, hidden)[0]  # index 0 obtains all hidden states
            out = self.out_layer(latent_seqs)
            outputs.append(out)
            hidden = latent_seqs.clone().squeeze(1).unsqueeze(0)
            # select input for next iteration
            if self.training and random.random() < teacher_forcing_ratio:
                if ys_batch is None:
                    print('ys_batch is required as input for decoder during training')
                    quit()
                # teacher forcing
                inputs = ys_batch[:, k]
                inputs = inputs.reshape(-1, 1, 1)
            else:
                inputs = out
            inputs = inputs
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs
