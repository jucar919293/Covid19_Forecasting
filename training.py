import torch
from epiweeks import Week
from modulesRNN.utils import Dataset, trainingModel
from seq2seqModels.models import Encoder2Decoder, Input2EncoderDecoder, InputEncoderDecoder, EncoderDecoder
import matplotlib.pyplot as plt

device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_type = torch.float32

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
# noinspection DuplicatedCode
weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]

weeks = [Week.fromstring(y) for y in weeks_strings]

# regions = ['X', 'AL', 'AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY'
#            'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH'
#            'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']
# regions = ['X']

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

# Path de figs
path_figs = './trainedModels/Windowed/Figs/'


# Main function
# noinspection PyPep8Naming
def training_process():
    last_week_data = Week.fromstring('202106')  # Total weeks: 49
    model_path_save = model_path[1]
    T = 10
    stride = 1
    total_weeks = 49
    n_min_seqs = 10  # Goes from 1 to 31(totalWeeks - T + stride / stride) - weakAhead
    max_val_week = total_weeks - wk_ahead + 1
    min_val_week = T + n_min_seqs - 1

    print("Initializing ...")

    for region in regions:

        print(f'Region: {region}')
        total_data_seq = Dataset(data_path, last_week_data, region, include_col, wk_ahead)
        _, ysT, _, _, allyT = total_data_seq.create_seqs_limited(T, stride, RNN_DIM, get_test=False)
        ysT = total_data_seq.scale_back_Y(ysT)
        yT = total_data_seq.scale_back_Y(allyT[-1])

        for ew, ew_str in zip(weeks[min_val_week:max_val_week], weeks_strings[min_val_week-1:max_val_week-1]):
            print(f'Week:{ew}')
            fig, ax = plt.subplots()
            twin1 = ax.twinx()
            dataset = Dataset(data_path, ew, region, include_col, wk_ahead)
            seqs, ys, mask_seq, mask_ys, allys, test = dataset.create_seqs_limited(T, stride, RNN_DIM, get_test=True)
            # Creating seq2seqModel
            seqModel = InputEncoderDecoder(T, seqs.shape[-1], RNN_DIM, wk_ahead)

            # Change device
            # Trainig process
            val, loss = trainingModel(seqModel, dataset,
                                      0.001, 500,
                                      seqs, mask_seq, ys, ysT, mask_ys, allys,
                                      two_encoder=False,
                                      get_att=False)
            total_epoch1 = len(val)
            twin1.plot(range(total_epoch1), val, c='orange')
            twin1.plot([])
            ax.plot(range(total_epoch1), loss, c='blue')
            # fig.show()
            val, loss = trainingModel(seqModel, dataset,
                                      0.0001, 500,
                                      seqs, mask_seq, ys, ysT, mask_ys, allys,
                                      two_encoder=False,
                                      get_att=False)
            # Ploting loss and eval
            total_epoch2 = len(val) - 1
            twin1.plot(range(total_epoch1-1, total_epoch1 + total_epoch2), val, c='orange')
            twin1.plot([])
            ax.plot(range(total_epoch1-1, total_epoch1 + total_epoch2), loss, c='blue')
            twin1.legend(['Validation', 'Loss'])

            val, loss = trainingModel(seqModel, dataset,
                                      0.00001, 500,
                                      seqs, mask_seq, ys, ysT, mask_ys, allys,
                                      two_encoder=False,
                                      get_att=False)
            # Ploting loss and eval
            total_epoch3 = len(val) - 1
            twin1.plot(range(total_epoch1 + total_epoch2-1, total_epoch1+total_epoch2+total_epoch3), val, c='orange')
            twin1.plot([])
            ax.plot(range(total_epoch1 + total_epoch2-1, total_epoch1+total_epoch2+total_epoch3), loss, c='blue')
            twin1.legend(['Validation', 'Loss'])
            # fig.show()
            # plt.close()
            torch.cuda.empty_cache()

        # Saving the model and plot
            path_model = model_path_save + region + '_' + ew_str + '_' + '.pth'
            torch.save(seqModel.state_dict(), path_model)
            name_image = region + '_' + ew_str + '_' + 'InputED' + '.png'
            fig.savefig(path_figs + name_image)
        # plt.show()


# Function to be executed.
if __name__ == '__main__':
    training_process()
