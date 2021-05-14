import torch
from epiweeks import Week
from modulesRNN.utils import Dataset, trainingModel
from seq2seqModels.models import InputEncoderDecoder,\
                                 EncoderDecoder,\
                                 EncoderDecoderHidden,\
                                 InputEncoderDecoderHidden,\
                                 EncoderAttentionDecoder,\
                                 InputEncoderAttentionDecoder,\
                                 InputEncoderv2Decoder, Input2EncoderDecoder

import matplotlib.pyplot as plt
from numpy import array
import modulesRNN.utils as utils
import pandas as pd
import gc


device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_type = torch.float32

"""
    User input parameters
"""
# Introduce the path were the data is storage
data_path = './data/train_data_weekly_vEW202105.csv'
model_path_or = './trainedModels/'
version_models = ['SimpleForm/', 'Windowed/', 'WindowedHidden/', 'WindowedTemporal/', 'Tests/']
aproaches = ['ED', 'InputED']

version_model = version_models[4]
aproach = aproaches[0]
model_path_save = model_path_or + version_model + aproach + '/'
# Path de figs
path_figs = model_path_or + version_model + 'Figs/'

# Select future target
wk_ahead = 4
# Prediction week (last week of data loaded)
# noinspection DuplicatedCode
weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]

weeks = [Week.fromstring(y) for y in weeks_strings]

# regions = ['X', 'AL', 'AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY'
#            'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH'
#            'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']
# regions = ['X', 'TX', 'GA', 'LA', 'MO']
regions = ['X']

# Select signals
include_col = ['target_death', 'retail_and_recreation_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline',
               'parks_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline',
               'workplaces_percent_change_from_baseline',
               'residential_percent_change_from_baseline',
               'apple_mobility',
               'dex', 'dex_a',
               'dex_income_1',
               'dex_income_1_a',
               'dex_income_2',
               'dex_income_2_a',
               'dex_income_3',
               'dex_income_3_a',
               'dex_income_4',
               'dex_income_4_a',
               'dex_education_1',
               'dex_education_1_a',
               'dex_education_2',
               'dex_education_2_a',
               'dex_education_3',
               'dex_education_3_a',
               'dex_education_4',
               'dex_education_4_a',
               'dex_race_asian',
               'dex_race_asian_a',
               'dex_race_black',
               'dex_race_black_a',
               'dex_race_hispanic',
               'dex_race_hispanic_a',
               'dex_race_white',
               'dex_race_white_a',
               'people_total',
               'people_total_2nd_dose',
               'covidnet',
               'positiveIncrease',
               'negativeIncrease',
               'totalTestResultsIncrease',
               'onVentilatorCurrently',
               'inIcuCurrently'	,
               'recovered',
               'hospitalizedIncrease',
               'Observed Number',
               'Excess Higher Estimate',
               'death_jhu_incidence',
               'fb_survey_cli',
               'google_survey_cli',
               'fb_survey_wili',
               'Number of Facilities Reporting',
               'CLI Percent of Total Visits']


# Dim of rnn hidden states
RNN_DIM = 128

# Number of external signals
n_signals = len(include_col) - 1


# noinspection DuplicatedCode
def testing():
    last_week_data = Week.fromstring('202106')  # Total weeks: 49
    T = 10
    stride = 1
    total_weeks = 49
    n_min_seqs = 10  # Goes from 5 to 31(totalWeeks - T + stride / stride) - weakAhead
    max_val_week = total_weeks - wk_ahead + 1
    min_val_week = T + n_min_seqs - 1

    print("Initializing ...")
    print(device)
    for region in regions:
        preds = []
        rels = []
        mae = []
        mape = []
        rmse = []

        print(f'Region: {region}')
        total_data_seq = Dataset(data_path, last_week_data, region, include_col, wk_ahead)
        _, ysT, _, _, allyT = total_data_seq.create_seqs_limited(T, stride, RNN_DIM, get_test=False)
        # _, ysT, _, _, allyT = total_data_seq.create_seqs(n_min_seqs, RNN_DIM)
        ysT = total_data_seq.scale_back_Y(ysT)
        yT = total_data_seq.scale_back_Y(allyT[-1])

        for ew, ew_str in zip(weeks[min_val_week:max_val_week], weeks_strings[min_val_week-1:max_val_week-1]):
            # print(f'Week:{ew}')
            fig, ax = plt.subplots(num=1, clear=True)
            fig.subplots_adjust(right=0.80)
            twin1 = ax.twinx()
            twin2 = ax.twinx()
            twin2.spines["right"].set_position(("axes", 1.13))
            dataset = Dataset(data_path, ew, region, include_col, wk_ahead)
            # seqs, ys, mask_seq, mask_ys, allys = dataset.create_seqs(n_min_seqs, RNN_DIM)
            seqs, ys, mask_seq, mask_ys, allys, test = dataset.create_seqs_limited(T, stride, RNN_DIM, get_test=True)

            # allys_seq = allys.clone()
            # seqs = torch.cat([seqs, allys_seq.unsqueeze(-1)], -1)

            # allys = dataset.scale_back_Y(allys)
            # Creating seq2seqModel
            path_model = model_path_save + region + '_' + ew_str + '_' + '.pth'

            seqModel = InputEncoderDecoderHidden(T, seqs.shape[-1], RNN_DIM, wk_ahead)

            seqModel.load_state_dict(torch.load(path_model))
            seqModel.eval()
            predictions = seqModel(seqs[-1].unsqueeze(0), mask_seq[-1],
                                   allys[-1].unsqueeze(0))
            predictions = dataset.scale_back_Y(predictions)
            preds.append(predictions.cpu().detach().numpy())
            real = ysT[ys.shape[0]-1]
            rels.append(real.cpu().detach())
            mae.append(utils.mae_calc(real, predictions))
            mape.append(utils.mape_calc(real, predictions))
            rmse.append(utils.rmse_calc(real, predictions))

            # print(predictions)
        mae = torch.stack(mae)
        mae_p = mae.mean()
        mape = torch.stack(mape)
        mape_p = mape.mean()
        rmse = torch.stack(rmse)
        rmse_p = rmse.mean()

        # preds = torch.stack(preds, 1)
        rels = torch.stack(rels, 1).numpy()

        i = 0
        a = len(preds)
        for signal in preds:
            ax.plot(range(i, wk_ahead+i), signal)
            ax.plot(range(ys.shape[0] - n_min_seqs+1), rels[0], c='b')
            ax.plot(range(1, ys.shape[0] - n_min_seqs+1 + 1), rels[1], c='b')
            ax.plot(range(2, ys.shape[0] - n_min_seqs+1 + 2), rels[2], c='b')
            ax.plot(range(3, ys.shape[0] - n_min_seqs+1 + 3), rels[3], c='b')
            i += 1
        ax.set_ylim(0, 30000)
        plt.show()
        print(mae)
        print(mape)
        print(rmse)
        print(f'MAE:{mae_p} MAPE:{mape_p} RMSE:{rmse_p}')


if __name__ == '__main__':
    testing()
