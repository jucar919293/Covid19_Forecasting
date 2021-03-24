import torch
import pandas as pd
from epiweeks import Week
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

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
        self.testY = torch.tensor(self.y[len(self.y) - wk_ahead:len(self.y) + 1]).unsqueeze(0).type(dtype).to(device)

    # noinspection PyPep8Naming,PyTypeChecker
    def scale_back_Y(self, y):
        mean = torch.tensor([self.meanY]).type(dtype).to(device)
        std = torch.tensor([self.stdY]).type(dtype).to(device)
        return (y * std) + mean

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

        seqs = pad_sequence(seqs, batch_first=True).type(dtype).to(device)
        ys = pad_sequence(targets, batch_first=True).type(dtype).to(device)
        mask_seq = pad_sequence(mask_seq, batch_first=True).type(dtype).to(device)
        mask_ys = pad_sequence(mask_ys, batch_first=True).type(dtype).to(device)

        return seqs, ys, mask_seq, mask_ys
