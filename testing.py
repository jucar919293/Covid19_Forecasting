import torch
from epiweeks import Week
from modulesRNN.utils import Dataset
from seq2seqModels import encoder_decoder, two_encoder_decoder, inputAttention2ED, inputAttentionED

device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Introduce the path were the data is storage
data_path = './data/train_data_weekly_vEW202105.csv'
wk_ahead = 4
regions = ['TX', 'PA', 'LA', 'IL', 'GA', 'FL']
weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]
weeks = [Week.fromstring(y) for y in weeks_strings]
include_col = ['target_death', 'retail_and_recreation_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
               'residential_percent_change_from_baseline', 'positiveIncrease', 'negativeIncrease',
               'totalTestResultsIncrease', 'onVentilatorCurrently', 'inIcuCurrently', 'recovered',
               'hospitalizedIncrease', 'death_jhu_incidence', 'dex_a', 'apple_mobility', 'CLI Percent of Total Visits',
               'fb_survey_cli']
RNN_DIM = 128
n_signals = len(include_col) - 1


def plot_signals():
    pass


def testing():
    pass


if __name__ == '__main__':
    testing()