import torch.nn as nn
import torch
from modulesRNN.rnn import Encoder, Decoder, EncoderAttention, DecoderHidden, DecoderAttention, EncoderAttentionv2, \
    Decoder4Out, DecoderAttentionv2, DecoderHidden4out

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
    Seq2seq model: Input Attention Encoder-Decoder model
"""


# noinspection PyAbstractClass,DuplicatedCode,PyTypeChecker,PyPep8Naming
class InputEncoderv2Decoder(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead):
        super(InputEncoderv2Decoder, self).__init__()

        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = EncoderAttentionv2(dim_seq_in=self.n_signals,
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
    Seq2seq model: encoder Hiden-decoder model
"""


class EncoderDecoderHidden(nn.Module):
    def __init__(self, n_signals, rnn_dim, wk_ahead):
        super(EncoderDecoderHidden, self).__init__()
        self.wk_ahead = wk_ahead
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = Encoder(dim_seq_in=self.n_signals,
                               rnn_out=rnn_dim,
                               n_layers=1,
                               bidirectional=False, ).to(device, data_type)

        self.decoder = DecoderHidden(dim_seq_in=rnn_dim+1,
                                     rnn_out=int(rnn_dim),
                                     n_layers=1,
                                     bidirectional=False,
                                     dim_out=1,).to(device, data_type)

    # noinspection PyPep8Naming,PyTypeChecker

    def forward(self, seqs, mask_seqs, allys, ys=None, **kwargs):
        # forward pass
        # Using encoder:
        c_vector = self.encoder(seqs, mask_seqs, all_hidden=True)
        # Using decoder:
        predictions = self.decoder(c_vector, self.wk_ahead, allys, ys=ys)
        return predictions


"""
    Seq2seq model: Encoder Temporal-Decoder model
"""


class EncoderAttentionDecoder(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead):
        super(EncoderAttentionDecoder, self).__init__()
        self.wk_ahead = wk_ahead
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = Encoder(dim_seq_in=self.n_signals,
                               rnn_out=rnn_dim,
                               n_layers=1,
                               bidirectional=False, ).to(device, data_type)

        self.decoder = DecoderAttention(dim_seq_in=rnn_dim+1,
                                        rnn_out=int(rnn_dim),
                                        n_layers=1,
                                        size_seq=size_seq,
                                        bidirectional=False,
                                        dim_out=1,).to(device, data_type)

    # noinspection PyPep8Naming,PyTypeChecker

    def forward(self, seqs, mask_seqs, allys, ys=None, **kwargs):
        # forward pass
        # Using encoder:
        c_vector = self.encoder(seqs, mask_seqs, all_hidden=True)
        # Using decoder:
        predictions = self.decoder(c_vector, self.wk_ahead, allys, ys=ys)
        return predictions


"""
    Seq2seq model: Input-encoder Hiden-decoder model
"""


class InputEncoderAttentionDecoder(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead):
        super(InputEncoderAttentionDecoder, self).__init__()

        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = EncoderAttentionv2(dim_seq_in=self.n_signals,
                                          rnn_out=rnn_dim,
                                          n_layers=1,
                                          size_seq=self.size_seq,
                                          bidirectional=False, ).to(device, data_type)

        self.decoder = DecoderAttentionv2(dim_seq_in=rnn_dim+1,
                                          rnn_out=int(rnn_dim),
                                          n_layers=1,
                                          size_seq=rnn_dim,
                                          bidirectional=False,
                                          dim_out=1,
                                          T = self.size_seq).to(device, data_type)

    def forward(self, seqs, mask_seq, allys, ys=None, get_att=False):

        # forward pass
        # Using encoder:
        c_vector, e_values, attention_values = self.encoder(seqs, mask_seq, get_att=get_att, all_hidden=True)
        # Using decoder:
        predictions = self.decoder(c_vector, self.wk_ahead, allys, ys=ys)
        if get_att:
            return predictions, e_values, attention_values
        else:
            return predictions


"""
    Seq2seq model: Input-encoder Hiden-decoder model
"""


class InputEncoderDecoderHidden(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead):
        super(InputEncoderDecoderHidden, self).__init__()

        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = EncoderAttentionv2(dim_seq_in=self.n_signals,
                                          rnn_out=rnn_dim,
                                          n_layers=1,
                                          size_seq=self.size_seq,
                                          bidirectional=False, ).to(device, data_type)

        self.decoder = DecoderHidden(dim_seq_in=rnn_dim+1,
                                     rnn_out=int(rnn_dim),
                                     n_layers=1,
                                     bidirectional=False,
                                     dim_out=1, ).to(device, data_type)

    def forward(self, seqs, mask_seq, allys, ys=None, get_att=False):

        # forward pass
        # Using encoder:
        c_vector, e_values, attention_values = self.encoder(seqs, mask_seq, get_att=get_att, all_hidden=True)
        # Using decoder:
        predictions = self.decoder(c_vector, self.wk_ahead, allys, ys=ys)
        if get_att:
            return predictions, e_values, attention_values
        else:
            return predictions


class InputEncoderDecoderHidden4(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead):
        super(InputEncoderDecoderHidden4, self).__init__()

        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        # Creating encoder:
        self.encoder = EncoderAttentionv2(dim_seq_in=self.n_signals,
                                          rnn_out=rnn_dim,
                                          n_layers=1,
                                          size_seq=self.size_seq,
                                          bidirectional=False, ).to(device, data_type)

        self.decoder = DecoderHidden4out(dim_seq_in=rnn_dim+1,
                                         rnn_out=int(rnn_dim),
                                         n_layers=1,
                                         bidirectional=False,
                                         dim_out=wk_ahead, ).to(device, data_type)

    def forward(self, seqs, mask_seq, allys, ys=None, get_att=False):

        # forward pass
        # Using encoder:
        c_vector, e_values, attention_values = self.encoder(seqs, mask_seq, get_att=get_att, all_hidden=True)
        # Using decoder:
        predictions = self.decoder(c_vector, self.wk_ahead, allys, ys=ys)
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
        self.encoder = EncoderAttentionv2(dim_seq_in=self.n_signals,
                                          rnn_out=rnn_dim,
                                          n_layers=1,
                                          size_seq=self.size_seq,
                                          bidirectional=False, ).to(device, data_type)

        self.encoder2 = Encoder(dim_seq_in=1,  # Change when more signals in seocnd encoder are included
                                rnn_out=rnn_dim,
                                n_layers=1,
                                bidirectional=False, ).to(device, data_type)

        self.decoder = Decoder(dim_seq_in=1,
                               rnn_out=int(rnn_dim),
                               n_layers=1,
                               bidirectional=False,
                               dim_out=1, ).to(device, data_type)

        self.bn1 = torch.nn.BatchNorm1d(num_features=rnn_dim, affine=True).to(device, data_type)

    def forward(self, seqs, mask_seqs, allys, ys=None, get_att=False):
        # forward pass
        # Using encoder:
        c_vector, e_values, attention_values = self.encoder(seqs, mask_seqs)
        c_vector2 = self.encoder2(allys.unsqueeze(-1), mask_seqs)
        # concat = torch.cat((c_vector, c_vector2), 1)
        sum_vec = self.bn1(c_vector.add(c_vector2))
        # Using decoder:
        predictions = self.decoder(sum_vec, self.wk_ahead, ys)
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
