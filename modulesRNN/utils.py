import torch
import pandas as pd
from epiweeks import Week
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time

data_type = torch.float64
"""
    Useful functions
"""


def convert_to_epiweek(x):
    return Week.fromstring(str(x))


# Defining Mean Absolute Error loss function
def mae_calc(pred, true):
    # Get absolute difference
    differences = pred - true
    absolute_differences = differences.abs()
    # Get the mean
    mean_absolute_error = absolute_differences.mean()
    return mean_absolute_error


# Defining Mean Squared Error loss function
def mse_calc(pred, true):
    # Get squared difference
    differences = pred - true
    squared_differences = differences ** 2
    # Get the mean
    mean_squared_error = squared_differences.mean()
    return mean_squared_error


# Defining Mean Absolute Percentage Error loss function
def mape_calc(pred, true):
    # Get squared difference
    differences = pred - true
    absolute_differences_divided = differences / true
    absolute_differences_divided = absolute_differences_divided.abs()
    # Get the mean
    mean_absolute_percentage = absolute_differences_divided.mean() * 100
    return mean_absolute_percentage


def rmse_calc(pred, true):
    # Get squared difference
    differences = pred - true
    squared_differences = differences ** 2
    # Get the mean
    root_mean_squared_error = (squared_differences.mean().sqrt())
    return root_mean_squared_error


"""
    Data preprocessing
"""


# noinspection DuplicatedCode
class Dataset:
    def __init__(self, data_path, last_epi_week, region, include_col, wk_ahead):
        self.wk_ahead = wk_ahead
        self.df = pd.read_csv(data_path)
        self.df['epiweek'] = self.df.loc[:, 'epiweek'].apply(convert_to_epiweek)

        # subset data using init parameters
        self.df = self.df[(self.df["epiweek"] < last_epi_week) & (self.df["region"] == region)]
        self.df = self.df[include_col]

        # get data as array
        self.x = self.df.iloc[:, 1:].values
        self.y = self.df.loc[:, 'target_death'].values

        # scale Y
        self.meanY = self.y.mean()
        self.stdY = self.y.std()
        self.scaledY = self.y - self.meanY
        self.scaledY /= self.stdY
        self.testY = torch.tensor(self.y[len(self.y) - wk_ahead:len(self.y) + 1], dtype=data_type).unsqueeze(0)

    # noinspection PyPep8Naming,PyTypeChecker
    def scale_back_Y(self, y):
        mean = torch.tensor([self.meanY], dtype=data_type)
        std = torch.tensor([self.stdY], dtype=data_type)
        return (y * std) + mean

    # noinspection DuplicatedCode
    def create_seqs_limited(self, t, stride, rnn_dim, get_test=False):
        # convert to small sequences for training, all length T
        seqs = []
        targets = []
        mask_seq = []
        mask_ys = []
        allys = []
        num_seqs = (self.x.shape[0]-t+stride) // stride
        for n in range(num_seqs):  # x.shape: [total_size_data, number_signals]
            seqs.append(torch.tensor(self.x[n*stride:n*stride+t, :]))
            last_data = t + stride*n - 1
            y_ = self.scaledY[last_data + 1:last_data + 1 + self.wk_ahead]
            targets.append(torch.tensor(y_))
            # Mask Sequences
            m_seq = torch.zeros((t, rnn_dim))  # NOTE: this is useful for temporal attention
            m_seq[-1] = 1
            mask_seq.append(m_seq)
            # Mask targets
            mask_ys.append(torch.ones(len(y_)))
            # All sequence
            ally_ = self.scaledY[n*stride:n*stride+t]
            allys.append(torch.tensor(ally_))

        seqs = pad_sequence(seqs, batch_first=True).double()
        ys = pad_sequence(targets, batch_first=True).double()
        mask_seq = pad_sequence(mask_seq, batch_first=True).double()
        mask_ys = pad_sequence(mask_ys, batch_first=True).double()
        allys = pad_sequence(allys, batch_first=True).double()
        test = seqs[num_seqs-1].unsqueeze(0).double()
        if get_test:
            return seqs, ys, mask_seq, mask_ys, allys, test
        else:
            return seqs, ys, mask_seq, mask_ys, allys

    def create_seqs(self, min_len_size, rnn_dim):
        """
        Manipulate data to make it suitable for RNN
        We want to use fixed length sequences and fixed length output
        We construct masks to allow this
        """
        # convert to small sequences for training, starting with length 10
        seqs = []
        targets = []
        mask_seq = []
        mask_ys = []
        allys = []
        for length in range(min_len_size, self.df.shape[0] + 1):
            # Sequences
            seqs.append(torch.from_numpy(self.x[:length, :]))
            # Targets
            y_ = self.scaledY[length:length + self.wk_ahead]
            targets.append(torch.from_numpy(y_))
            # Mask Sequences
            m_seq = torch.zeros((length, rnn_dim))  # NOTE: this is useful for temporal attention
            m_seq[-1] = 1
            mask_seq.append(m_seq)
            # Mask targets
            mask_ys.append(torch.ones((len(y_))))
            # All sequence
            ally_ = self.scaledY[:length]
            allys.append(torch.from_numpy(ally_))

        seqs = pad_sequence(seqs, batch_first=True).double()
        ys = pad_sequence(targets, batch_first=True).double()
        mask_seq = pad_sequence(mask_seq, batch_first=True).double()
        mask_ys = pad_sequence(mask_ys, batch_first=True).double()
        allys = pad_sequence(allys,batch_first=True).double()

        return seqs, ys, mask_seq, mask_ys, allys


# noinspection PyPep8Naming,DuplicatedCode
def trainingModel(seq2seqmodel, lr, epochs, seqs, mask_seq, ys, ysT, mask_ys, allys):
    N = seqs.shape[0]
    mini_batch_size = N // 2
    print(f'Total batch: {N}')
    print(f'Mini batch: {mini_batch_size}')
    params = list(seq2seqmodel.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr)
    start_time = time.time()

    for epoch in range(epochs):
        seq2seqmodel.train()
        for _ in range(N // mini_batch_size):
            # get batch of data
            idx = np.random.choice(N, mini_batch_size)
            seqs_batch = seqs[idx, :]
            ys_batch = ys[idx, :]
            mask_seq_batch = mask_seq[idx, :]
            mask_ys_batch = mask_ys[idx, :]
            predictions = seq2seqmodel.forward(seqs_batch, mask_seq_batch, ys_batch, get_att=False)
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

        # TODO: Implement a early stop in training calculating the error in testing
        if epoch % testing_epoch_print == 0:
            seq2seqmodel.eval()
            predictions = seq2seqmodel.forward(seqs, mask_seq, ys, get_att=False)
            print("Test Process Eval")
            print(seq2seqmodel.primer_dataset.scale_back_Y(predictions))
            print(ysT)
            elapsed = time.time() - start_time
            pred_loss = F.mse_loss(predictions, ysT, reduction='none')
            pred_loss = pred_loss.mean()
            print('Epoch: %d, Loss: %.3e, Time: %.3f, Learning Rate: %.1e'
                  % (epoch, pred_loss.item(), elapsed, lr))
            start_time = time.time()