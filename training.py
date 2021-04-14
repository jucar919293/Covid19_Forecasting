import torch
from epiweeks import Week
from modulesRNN.utils import Dataset
from seq2seqModels.models import Encoder2Decoder, Input2EncoderDecoder, InputEncoderDecoder, EncoderDecoder

device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
    User input parameters
"""
# Introduce the path were the data is storage
data_path = './data/train_data_weekly_vEW202105.csv'
model_path = ['./trainedModels/Windowed/Input2ED/',
              './trainedModels/Windowed/InputED/',
              './trainedModels/Windowed/2ED/',
              './trainedModels/Windowed/ED/']

# Select future target
wk_ahead = 4
# Prediction week (last week of data loaded)
weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]

weeks = [Week.fromstring(y) for y in weeks_strings]

# regions = ['X', 'AL', 'AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY'
#            'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH'
#            'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']
regions = ['X']

# Select signals
include_col = ['target_death', 'retail_and_recreation_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
               'residential_percent_change_from_baseline', 'positiveIncrease', 'negativeIncrease',
               'totalTestResultsIncrease', 'onVentilatorCurrently', 'inIcuCurrently', 'recovered',
               'hospitalizedIncrease', 'death_jhu_incidence', 'dex_a', 'apple_mobility', 'CLI Percent of Total Visits',
               'fb_survey_cli']

# Dim of rnn hidden states
RNN_DIM = 128

# Number of external signals
n_signals = len(include_col) - 1


# Main function
# noinspection PyPep8Naming
def training_process():
    last_week_data = Week.fromstring('202106')  # Total weeks: 49
    model_path_save = model_path[3]
    T = 15
    stride = 1
    total_weeks = 49
    n_min_seqs = 29  # Goes from 1 to 30(totalWeeks - T + stride / stride)
    max_val_week = total_weeks - wk_ahead + 1
    min_val_week = T + n_min_seqs + 1
    print("Initializing ...")

    for region in regions:

        print(f'Region: {region}')
        total_data_seq = Dataset(data_path, last_week_data, region, include_col, wk_ahead)
        _, _, _, _, allyT = total_data_seq.create_seqs_limited(T, stride, RNN_DIM, get_test=False)
        yT = total_data_seq.scale_back_Y(allyT[-1])

        for ew, ew_str in zip(weeks[min_val_week:max_val_week], weeks_strings[min_val_week-1:max_val_week-1]):
            print(f'Week:{ew}')
            dataset = Dataset(data_path, ew, region, include_col, wk_ahead)
            seqs, ys, mask_seq, mask_ys, allys, test = dataset.create_seqs_limited(T, stride, RNN_DIM, get_test=True)

            # Creating seq2seqModel
            seqModel = EncoderDecoder(seqs.shape[-1], RNN_DIM, wk_ahead)

            # Testing the seqModel
            seqModel.eval()
            seqs, mask_seq = seqs.to(device), mask_seq.to(device)
            predictions = seqModel(seqs, mask_seq)

            # Saving the model
            path_model = model_path_save + region + '_' + ew_str + '_' + '.pth'
            torch.save(seqModel.state_dict(), path_model)


# Function to be executed.
if __name__ == '__main__':
    training_process()
