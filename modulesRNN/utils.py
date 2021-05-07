import torch
import pandas as pd
from epiweeks import Week
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_type = torch.float32
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
    torch.cuda.empty_cache()
    return mean_absolute_error


# Defining Mean Squared Error loss function
def mse_calc(pred, true):
    # Get squared difference
    differences = pred - true
    squared_differences = differences ** 2
    # Get the mean
    mean_squared_error = squared_differences.mean()
    torch.cuda.empty_cache()
    return mean_squared_error


# Defining Mean Absolute Percentage Error loss function
def mape_calc(pred, true, mask=None):
    # Get squared difference
    differences = pred - true
    absolute_differences_divided = differences / true
    absolute_differences_divided = absolute_differences_divided.abs()
    torch.cuda.empty_cache()

    if mask is not None:
        mask_absolute = absolute_differences_divided * mask
        mean_absolute_percentage = mask_absolute.mean() * 100
    else:
        mean_absolute_percentage = absolute_differences_divided.mean() * 100
    # Get the mean
    return mean_absolute_percentage


def rmse_calc(pred, true):
    # Get squared difference
    differences = pred - true
    squared_differences = differences ** 2
    # Get the mean
    root_mean_squared_error = (squared_differences.mean().sqrt())
    torch.cuda.empty_cache()
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
        self.df = self.df[(self.df["epiweeko"] < last_epi_week) & (self.df["region"] == region)]
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
        mean = torch.tensor([self.meanY], dtype=data_type).to(device)
        std = torch.tensor([self.stdY], dtype=data_type).to(device)
        torch.cuda.empty_cache()
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

        seqs = pad_sequence(seqs, batch_first=True).to(device, data_type)
        ys = pad_sequence(targets, batch_first=True).to(device, data_type)
        mask_seq = pad_sequence(mask_seq, batch_first=True).to(device, data_type)
        mask_ys = pad_sequence(mask_ys, batch_first=True).to(device, data_type)
        allys = pad_sequence(allys, batch_first=True).to(device, data_type)
        test = seqs[num_seqs-1].unsqueeze(0).to(device, data_type)
        if get_test:
            return seqs, ys, mask_seq, mask_ys, allys, test
        else:
            return seqs, ys, mask_seq, mask_ys, allys

    # TODO: Create seqs that only take some past values, not all
    def create_seqs_limited2(self, t, stride, rnn_dim, get_test=False):
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

        seqs = pad_sequence(seqs, batch_first=True).to(device, data_type)
        ys = pad_sequence(targets, batch_first=True).to(device, data_type)
        mask_seq = pad_sequence(mask_seq, batch_first=True).to(device, data_type)
        mask_ys = pad_sequence(mask_ys, batch_first=True).to(device, data_type)
        allys = pad_sequence(allys, batch_first=True).to(device, data_type)
        test = seqs[num_seqs-1].unsqueeze(0).to(device, data_type)
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

        seqs = pad_sequence(seqs, batch_first=True).to(device, data_type)
        ys = pad_sequence(targets, batch_first=True).to(device, data_type)
        mask_seq = pad_sequence(mask_seq, batch_first=True).to(device, data_type)
        mask_ys = pad_sequence(mask_ys, batch_first=True).to(device, data_type)
        allys = pad_sequence(allys,batch_first=True).to(device, data_type)

        return seqs, ys, mask_seq, mask_ys, allys


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0.0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


# noinspection PyPep8Naming
def trainingModel(seq2seqmodel, dataset,
                  lr, epochs, min_delta, patience,
                  seqs, mask_seq, ys, mask_ys, allys, ysT,
                  allys_needed=False, get_att=False
                  ):
    loss = []
    val = []
    test = []
    training_epoch_print = 20
    testing_epoch_print = 20
    n_batch = seqs.shape[0]
    mini_batch_size = n_batch // 2
    print(f'Total Nº batch: {n_batch}')
    print(f' Nº mini batch: {mini_batch_size}')
    params = list(seq2seqmodel.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr)
    start_time = time.time()
    stop = EarlyStopping(mode='min', min_delta=min_delta, patience=patience, percentage=False)
    for epoch in range(epochs+1):
        seq2seqmodel.train()
        for _ in range(n_batch // mini_batch_size):
            # get batch of data
            idx = np.random.choice(n_batch, mini_batch_size)
            seqs_batch = seqs[idx, :]
            ys_batch = ys[idx, :]
            mask_seq_batch = mask_seq[idx, :]
            mask_ys_batch = mask_ys[idx, :]
            allys_batch = allys[idx, :]
            # seqs, mask_seqs, allys, ys=None, get_att=False
            if allys_needed:
                predictions = seq2seqmodel(seqs_batch, mask_seq_batch, allys_batch, ys_batch, get_att=get_att)
            else:
                predictions = seq2seqmodel(seqs_batch, mask_seq_batch, ys_batch, get_att=get_att)

            # prediction loss
            pred_loss = F.mse_loss(predictions, ys_batch, reduction='none') * mask_ys_batch
            pred_loss = pred_loss.mean()
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()

        # TODO: Improve graphs of training
        if epoch % training_epoch_print == 0:
            # print("Training Process Eval")
            # noinspection PyUnboundLocalVariable
            loss_send = pred_loss.item()
            loss.append(loss_send)
            # print(f'Epoch: {epoch:d}, Learning Rate: {lr:.1e}')
            # loss.append(pred_loss.item())
        if epoch % testing_epoch_print == 0:
            seq2seqmodel.eval()
            if allys_needed:
                predictions = seq2seqmodel(seqs, mask_seq, allys, ys, get_att=get_att)
            else:
                predictions = seq2seqmodel(seqs, mask_seq, get_att=get_att)
            # print("Test Process Eval")
            # print("Predictions:")
            # print(dataset.scale_back_Y(predictions))
            # print("Real Values:")
            # print(ysT[:ys.shape[0]])
            elapsed = time.time() - start_time
            # val_loss = mape_calc(dataset.scale_back_Y(predictions), ysT[:ys.shape[0]]) / (4*predictions.shape[0])
            test_loss = (mape_calc(dataset.scale_back_Y(predictions), ysT[:ys.shape[0]], mask_ys))
            test_loss = (test_loss * ys.shape[0]*4) / ((ys.shape[0]*4)-10)
            test.append(test_loss.to(torch.device("cpu")))
            val_loss = (mape_calc(dataset.scale_back_Y(predictions), ysT[:ys.shape[0]], abs(mask_ys-1)))
            val_loss = (val_loss * ys.shape[0]*4) / 4
            val_loss = val_loss.to(torch.device("cpu"))
            val.append(val_loss)
            have_to_stop = stop.step(test_loss)
            # print(stop.best)
            if have_to_stop:
                print(f'Epoch: {epoch:d}, Learning Rate: {lr:.1e}')
                print(dataset.scale_back_Y(predictions))
                # print("Real Values:")
                print(ysT[:ys.shape[0]])
                break
            # print("Testing Process Eval")
            # print('Epoch: %d, Validation: %.4f, Time: %.3f, Learning Rate: %.1e'
            #      % (epoch, val_loss, elapsed, lr))
            start_time = time.time()
    return val, loss, test
