from abc import ABC
from typing import Any
import pandas as pd
import torch
import torch.nn as nn
from .weight_init import weight_init
import random
from torch.nn.utils.rnn import pad_sequence
"""
Classes used in the architectures
"""
device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_type = torch.float32


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
        torch.cuda.empty_cache()
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
        self.attention_total = {}
        self.e_values_total = {}
        # Creating Attention
        self.attention = InputAttention(size_hidden=self.rnn_out,
                                        n_features=self.dim_seq_in,
                                        size_seq=self.size_seq)

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            num_layers=n_layers,
            batch_first=True,
        )

        # init weights
        weight_init(self.rnn)

    def forward(self, seqs, mask=None, get_att=None, all_hidden=False):
        """
        param mask: binary tensor with same shape as seqs
        """

        hidden_total = []

        seqs_update = seqs.clone()
        test_h = self.rnn(seqs[:, 0, :].unsqueeze(1))[0]
        h_t_previous = torch.zeros_like(test_h)
        # h_t_previous = torch.zeros((seqs.shape[0], self.rnn_out)).unsqueeze(1).to(device, data_type)
        signal_k = seqs.transpose(1, 2)

        for t in range(seqs.shape[1]):
            h_t_previous_signals = h_t_previous.repeat(1, seqs.shape[-1], 1)
            # Obtaining attention values for time t
            e_values_t, att_values_t = self.attention(signal_k, h_t_previous_signals)
            seq_t = att_values_t * seqs[:, t, :]
            seqs_update[:, t, :] = seq_t
            torch.cuda.empty_cache()
            # Updating values for next step t+1
            h_t = self.rnn(seq_t.unsqueeze(1), h_t_previous.squeeze(1).unsqueeze(0))[1]
            hidden_total.append(h_t.squeeze(0))
            h_t_previous = h_t.clone().squeeze(0).unsqueeze(1)

            self.e_values_total[str(t)] = e_values_t.squeeze(0).squeeze(-1).detach().cpu().numpy()
            self.attention_total[str(t)] = att_values_t.squeeze(0).squeeze(-1).detach().cpu().numpy()
        torch.cuda.empty_cache()
        hidden_total = torch.stack(hidden_total, 1)
        # Pass through first rnn
        if self.training and not get_att:
            torch.cuda.empty_cache()
            if all_hidden:
                latent_seqs = hidden_total
                return latent_seqs, self.e_values_total, self.attention_total
            else:
                latent_seqs = hidden_total * mask  # keep hidden states that correspond to non-zero in seqs
                latent_seq = latent_seqs.sum(1)
                return latent_seq, self.e_values_total, self.attention_total
        elif get_att:
            df_a = pd.DataFrame(self.attention_total)
            df_e = pd.DataFrame(self.e_values_total)
            torch.cuda.empty_cache()
            if all_hidden:
                latent_seqs = hidden_total
                return latent_seqs,  df_e, df_a
            else:
                latent_seqs = hidden_total * mask  # keep hidden states that correspond to non-zero in seqs
                latent_seq = latent_seqs.sum(1)
                return latent_seq,  df_e, df_a
        else:
            torch.cuda.empty_cache()
            if all_hidden:
                latent_seqs = hidden_total
                return latent_seqs, self.e_values_total, self.attention_total
            else:
                latent_seqs = hidden_total * mask  # keep hidden states that correspond to non-zero in seqs
                latent_seq = latent_seqs.sum(1)
                return latent_seq, self.e_values_total, self.attention_total
        # pdb.set_trace()
        # pdb.set_trace()


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

    def forward(self, seqs, mask, all_hidden=False):
        """
        param mask: binary tensor with same shape as seqs
        """
        # pass through first rnn

        latent_seqs = self.rnn(seqs)[0]  # index 0 obtains all hidden states
        # pdb.set_trace()
        if all_hidden:
            return latent_seqs
        else:
            latent_seqs = latent_seqs * mask  # keep hidden states that correspond to non-zero in seqs
            latent_seq = latent_seqs.sum(1)
            return latent_seq


# noinspection PyAbstractClass
class Decoder(nn.Module):
    def __init__(
            self,
            dim_seq_in: int = 5,
            rnn_out: int = 128,
            n_layers: int = 1,
            bidirectional: bool = False,
            dim_out: int = 1,
            dropout: float = 0.05,
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

    def forward(self, hidden, k_wk_ahead, ys=None, teacher_forcing_ratio=0.5):
        """
        Teacher forcing

        param k_wk_ahead: how many weeks ahead to forecast
        """
        inputs = torch.zeros((hidden.shape[0], 1, 1)).to(device, data_type)
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
                if ys is None:
                    print('ys_batch is required as input for decoder during training')
                    quit()
                # teacher forcing
                inputs = ys[:, k]
                inputs = inputs.reshape(-1, 1, 1)
            else:
                inputs = out
            inputs = inputs
        torch.cuda.empty_cache()
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs


# noinspection PyAbstractClass
class DecoderHidden(nn.Module):
    def __init__(
            self,
            dim_seq_in: int = 129,
            rnn_out: int = 128,
            n_layers: int = 1,
            bidirectional: bool = False,
            dim_out: int = 1,
            dropout: float = 0.05,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (hiden state size)
        param rnn_out: output dimension for rnn
        """
        super(DecoderHidden, self).__init__()

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

    def forward(self, hiddens, k_wk_ahead, allys, ys=None, teacher_forcing_ratio=0.5):
        """
        Teacher forcing

        param k_wk_ahead: how many weeks ahead to forecast
        """
        inputs = torch.cat((hiddens, allys.unsqueeze(-1)), dim=2)
        # pdb.set_trace()
        last_d = self.rnn(inputs)[1]
        # note that hidden should be of (num_layers * num_directions, batch, hidden_size)
        outputs = []

        for k in range(k_wk_ahead):
            # pass through first rnn
            out = self.out_layer(last_d)
            outputs.append(out.squeeze(0))
            # select input for next iteration
            if self.training and random.random() < teacher_forcing_ratio:
                if ys is None:
                    print('ys_batch is required as input for decoder during training')
                    quit()
                # teacher forcing
                input = torch.cat((hiddens[:, -1], ys[:, k].unsqueeze(-1)), dim=1)
            else:
                input = torch.cat((hiddens[:, -1], out.squeeze(0)), dim=1)
            input = input.unsqueeze(1)
            last_d = self.rnn(input, last_d)[1]  # index 0 obtains all hidden state
        torch.cuda.empty_cache()
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs


# noinspection PyAbstractClass
class DecoderAttention(nn.Module):
    def __init__(
            self,
            dim_seq_in: int = 129,
            rnn_out: int = 128,
            n_layers: int = 1,
            size_seq: int = 36,
            bidirectional: bool = False,
            dim_out: int = 1,
            dropout: float = 0.05,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (hiden state size)
        param rnn_out: output dimension for rnn
        """
        super(DecoderAttention, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.size_seq = size_seq

        hidden_size = self.rnn_out // 2 if self.bidirectional else self.rnn_out

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.attention = InputAttention(size_hidden=self.rnn_out,
                                        n_features=self.dim_seq_in,
                                        size_seq=self.size_seq)

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

    def forward(self, hiddens, k_wk_ahead, allys, ys=None, teacher_forcing_ratio=0.5):
        """
        Teacher forcing

        param k_wk_ahead: how many weeks ahead to forecast
        """
        # Calculating attention weights (c_vector)
        d_h_total = []
        c_vectors = torch.zeros_like(hiddens)
        d_t_previous = torch.zeros((hiddens.shape[0], self.rnn_out)).unsqueeze(1).to(device, data_type)
        h_signals_k = hiddens.clone()

        for t in range(hiddens.shape[1]):
            d_t_previous_signals = d_t_previous.repeat(1, hiddens.shape[1], 1)
            # Obtaining attention values for time t
            e_values_t, att_values_t = self.attention(h_signals_k, d_t_previous_signals)
            hiddens_pond = att_values_t.unsqueeze(-1) * hiddens
            hiddens_pond = hiddens_pond.sum(1)
            c_vectors[:, t, :] = hiddens_pond
            torch.cuda.empty_cache()
            # Updating values for next step t+1
            inputs = torch.cat((hiddens_pond
                                , allys[:, t].unsqueeze(-1)),
                               dim=1).unsqueeze(1)
            d_t = self.rnn(inputs, d_t_previous.squeeze(1).unsqueeze(0))[0]
            d_h_total.append(d_t.squeeze(1))
            d_t_previous = d_t.clone()

        d_h_total = torch.stack(d_h_total, 1)
        # pdb.set_trace()

        # note that hidden should be of (num_layers * num_directions, batch, hidden_size)
        outputs = []
        # pdb.set_trace()
        last_d = d_h_total[:, -1]

        for k in range(k_wk_ahead):
            # pass through first rnn
            out = self.out_layer(last_d)
            outputs.append(out)
            # select input for next iteration
            if self.training and random.random() < teacher_forcing_ratio:
                if ys is None:
                    print('ys_batch is required as input for decoder during training')
                    quit()
                # teacher forcing
                input = torch.cat((c_vectors[:, -1], ys[:, k].unsqueeze(-1)), dim=1)
            else:
                input = torch.cat((c_vectors[:, -1], out), dim=1)
            input = input.unsqueeze(1)
            last_d = self.rnn(input, last_d.clone().unsqueeze(0))[1].squeeze(0)  # index 0 obtains all hidden state
        torch.cuda.empty_cache()
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs
