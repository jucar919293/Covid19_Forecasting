import torch.nn as nn
import torch
import numpy as np
from modulesRNN.rnn import EncoderAttention, Decoder, Encoder
import time
import torch.nn.functional as F

"""
    Seq2seq model: encoder decoder model
"""
device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
training_epoch_print = 100
testing_epoch_print = 500


# noinspection PyAbstractClass,DuplicatedCode,PyTypeChecker,PyPep8Naming
class Seq2SeqModel(nn.Module):
    def __init__(self, size_seq, n_signals, rnn_dim, wk_ahead, dataset=None):
        super(Seq2SeqModel, self).__init__()

        self.wk_ahead = wk_ahead
        self.size_seq = size_seq
        self.n_signals = n_signals
        self.primer_dataset = dataset
        # Creating encoder:
        self.encoder = EncoderAttention(dim_seq_in=self.n_signals,
                                        rnn_out=rnn_dim,
                                        n_layers=1,
                                        size_seq=self.size_seq,
                                        bidirectional=False, ).to(device).type(dtype)

        self.encoder2 = Encoder(dim_seq_in=1,  # Change when more signals in seocnd encoder are included
                                rnn_out=rnn_dim,
                                n_layers=1,
                                bidirectional=False, ).to(device).type(dtype)

        self.decoder = Decoder(dim_seq_in=1,
                               rnn_out=int(rnn_dim * 2),
                               n_layers=1,
                               bidirectional=False,
                               dim_out=1, ).to(device).type(dtype)

    def forward(self, seqs, mask_seqs, allys, ys=None):
        # forward pass
        # Using encoder:
        c_vector = self.encoder(seqs, mask_seqs)
        c_vector2 = self.encoder2(allys.unsqueeze(-1), mask_seqs)
        concat = torch.cat((c_vector, c_vector2), 1)
        # Using decoder:
        predictions = self.decoder(concat, self.wk_ahead, ys)
        return predictions

    def trainingModel(self, lr, epochs, seqs, mask_seq, ys, ysT, mask_ys, allys):
        N = seqs.shape[0]
        mini_batch_size = N // 2
        print(f'Total batch: {N}')
        print(f'Mini batch: {mini_batch_size}')
        params = list(self.parameters())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr)
        start_time = time.time()
        # ones = torch.ones(mask_ys.shape[0], mask_ys.shape[1]).type(dtype).to(device)
        # mask_ys_invert = mask_ys - ones
        # mask_ys_invert = mask_ys_invert.abs()

        for epoch in range(epochs):
            self.train()
            for _ in range(N // mini_batch_size):
                # get batch of data
                idx = np.random.choice(N, mini_batch_size)
                seqs_batch = seqs[idx, :]
                ys_batch = ys[idx, :]
                mask_seq_batch = mask_seq[idx, :]
                mask_ys_batch = mask_ys[idx, :]
                allys_batch = allys[idx, :]
                predictions = self.forward(seqs_batch, mask_seq_batch, allys_batch, ys_batch)
                # prediction loss
                pred_loss = F.mse_loss(predictions, ys_batch, reduction='none') * mask_ys_batch
                pred_loss = pred_loss.mean()
                optimizer.zero_grad()
                pred_loss.backward()
                optimizer.step()

            if epoch % training_epoch_print == 0:
                print("Training Process Eval")
                # noinspection PyUnboundLocalVariable
                print(f'Epoch: {epoch:d}, Loss: {pred_loss.item():.3e}, Learning Rate: {lr:.1e}')
                start_time = time.time()

            # TODO: Implement a early stop in training calculating the error in testing
            if epoch % testing_epoch_print == 0:
                self.eval()
                predictions = self.forward(seqs, mask_seq, allys, ys)
                print("Test Process Eval")
                print(self.primer_dataset.scale_back_Y(predictions))
                print(ysT)
                elapsed = time.time() - start_time
                pred_loss = F.mse_loss(predictions, ysT, reduction='none')
                pred_loss = pred_loss.mean()
                print('Epoch: %d, Loss: %.3e, Time: %.3f, Learning Rate: %.1e'
                      % (epoch, pred_loss.item(), elapsed, lr))
                start_time = time.time()
