import torch
from epiweeks import Week
from modulesRNN.utils import Dataset
from seq2seqModels import encoder_decoder, two_encoder_decoder, inputAttention2ED, inputAttentionED

device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
    User input parameters
"""
# Introduce the path were the data is storage
data_path = './data/train_data_weekly_vEW202105.csv'
model_path = ['./trainedModels/Input2ED/',
              './trainedModels/InputED/',
              './trainedModels/2ED/',
              './trainedModels/ED/']

# Select future target
wk_ahead = 4
# Prediction week (last week of data loaded)
weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]

weeks = [Week.fromstring(y) for y in weeks_strings]

# regions = ['X', 'AL', 'AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY'
#            'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH'
#            'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']
regions = ['TX', 'PA', 'LA', 'IL', 'GA', 'FL']

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
    last_week_data = Week.fromstring('202106')
    model_path_save = model_path[0]
    min_len_sequence = 10
    number_models = 2
    print("Initializing ...")

    for i in range(number_models):
        for region in regions:

            print(f'Region: {region}')
            total_data_seq = Dataset(data_path, last_week_data, region, include_col, wk_ahead)
            _, ysT, _, _, _ = total_data_seq.create_seqs(min_len_sequence, RNN_DIM)
            ysT = total_data_seq.scale_back_Y(ysT)

            for ew, ew_str in zip(weeks[20:46], weeks_strings[19:45]):
                print(f'Week:{ew}')
                dataset = Dataset(data_path, ew, region, include_col, wk_ahead)
                seqs, ys, mask_seq, mask_ys, allys = dataset.create_seqs(min_len_sequence, RNN_DIM)

                # Creating seq2seqModel
                seqModel = two_encoder_decoder.Seq2SeqModel(seqs.shape[1], seqs.shape[-1], dataset, RNN_DIM, wk_ahead)
                seqModel.trainingModel(0.001, 3000, seqs, mask_seq, ys, ysT[:ys.shape[0], :], mask_ys, allys)
                seqModel.trainingModel(0.0001, 3000, seqs, mask_seq, ys, ysT[:ys.shape[0], :], mask_ys, allys)
                seqModel.trainingModel(0.00001, 1500, seqs, mask_seq, ys, ysT[:ys.shape[0], :], mask_ys, allys)

                # Saving the model
                path_model = model_path_save + region + '_' + ew_str + '_' + str(i) + '.pth'
                torch.save(seqModel.state_dict(), path_model)


# Function to be executed.
if __name__ == '__main__':
    training_process()
