import torch.nn as nn
import torch
from modulesRNN.rnn import Encoder, Decoder, EncoderAttention

device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_type = torch.float32
"""
    Seq2seq model: encoder decoder model
"""


# noinspection PyAbstractClass
class EncoderDecoder(nn.Module):
    def __init__(self, n_signals, rnn_dim, wk_ahead):
        super(EncoderDecoder, self).__init__()
        self.wk_ahead = wk_ahead
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = Encoder(dim_seq_in=self.n_signals,
                               rnn_out=rnn_dim,
                               n_layers=1,
                               bidirectional=False, ).to(device, data_type)

        self.decoder = Decoder(dim_seq_in=1,
                               rnn_out=int(rnn_dim),
                               n_layers=1,
                               bidirectional=False,
                               dim_out=1,).to(device, data_type)

    # noinspection PyPep8Naming,PyTypeChecker

    def forward(self, seqs, mask_seqs, ys=None, **kwargs):
        # forward pass
        # Using encoder:
        c_vector = self.encoder(seqs, mask_seqs)
        # Using decoder:
        predictions = self.decoder(c_vector, self.wk_ahead, ys=ys)
        return predictions


"""
    Seq2seq model: Input Attention Encoder-Decoder model
"""


# noinspection PyAbstractClass,DuplicatedCode,PyTypeChecker,PyPep8Naming
class InputEncoderDecoder(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead):
        super(InputEncoderDecoder, self).__init__()

        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = EncoderAttention(dim_seq_in=self.n_signals,
                                        rnn_out=rnn_dim,
                                        n_layers=1,
                                        size_seq=self.size_seq,
                                        bidirectional=False, ).to(device, data_type)

        self.decoder = Decoder(dim_seq_in=1,
                               rnn_out=int(rnn_dim),
                               n_layers=1,
                               bidirectional=False,
                               dim_out=1, ).to(device, data_type)

    def forward(self, seqs, mask_seq, ys=None, get_att=False):

        # forward pass
        # Using encoder:
        c_vector, e_values, attention_values = self.encoder(seqs, mask_seq, get_att=get_att)
        # Using decoder:
        predictions = self.decoder(c_vector, self.wk_ahead, ys=ys)
        if get_att:
            return predictions, e_values, attention_values
        else:
            return predictions


"""
    Seq2seq model: InputAttention 2EncoderDecoder
"""


# noinspection PyAbstractClass,DuplicatedCode,PyTypeChecker,PyPep8Naming
class Input2EncoderDecoder(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead, dataset=None):
        super(Input2EncoderDecoder, self).__init__()
        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        self.primer_dataset = dataset
        # Creating encoder:
        self.encoder = EncoderAttention(dim_seq_in=self.n_signals,
                                        rnn_out=rnn_dim,
                                        n_layers=1,
                                        size_seq=self.size_seq,
                                        bidirectional=False, )

        self.encoder2 = Encoder(dim_seq_in=1,  # Change when more signals in seocnd encoder are included
                                rnn_out=rnn_dim,
                                n_layers=1,
                                bidirectional=False, )

        self.decoder = Decoder(dim_seq_in=1,
                               rnn_out=int(rnn_dim * 2),
                               n_layers=1,
                               bidirectional=False,
                               dim_out=1, )

    def forward(self, seqs, mask_seqs, allys, ys=None, get_att=False):
        # forward pass
        # Using encoder:
        c_vector, e_values, attention_values = self.encoder(seqs, mask_seqs)
        c_vector2 = self.encoder2(allys.unsqueeze(-1), mask_seqs)
        concat = torch.cat((c_vector, c_vector2), 1)
        # Using decoder:
        predictions = self.decoder(concat, self.wk_ahead, ys)
        if get_att:
            return predictions, e_values, attention_values
        else:
            return predictions


"""
    Seq2seq model: Two Encoder Decoder
"""


# noinspection PyAbstractClass,DuplicatedCode,PyTypeChecker,PyPep8Naming
class Encoder2Decoder(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead, dataset=None):
        super(Encoder2Decoder, self).__init__()
        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        self.primer_dataset = dataset
        # Creating encoder:
        self.encoder = Encoder(dim_seq_in=self.n_signals,
                               rnn_out=rnn_dim,
                               n_layers=1,
                               bidirectional=False, )

        self.encoder2 = Encoder(dim_seq_in=1,  # Change when more signals in seocnd encoder are included
                                rnn_out=rnn_dim,
                                n_layers=1,
                                bidirectional=False, )

        self.decoder = Decoder(dim_seq_in=1,
                               rnn_out=int(rnn_dim * 2),
                               n_layers=1,
                               bidirectional=False,
                               dim_out=1, )

    def forward(self, seqs, mask_seq, allys, ys=None, **kwargs):
        # forward pass
        # Using encoder:
        c_vector = self.encoder(seqs, mask_seq)
        c_vector2 = self.encoder2(allys.unsqueeze(-1), mask_seq)
        concat = torch.cat((c_vector, c_vector2), 1)

        # Using decoder:
        predictions = self.decoder(concat, self.wk_ahead, ys)
        return predictions
