import torch
from epiweeks import Week
from modulesRNN.utils import Dataset, trainingModel
from seq2seqModels.models import InputEncoderDecoder,\
                                 EncoderDecoder,\
                                 EncoderDecoderHidden,\
                                 InputEncoderDecoderHidden,\
                                 EncoderAttentionDecoder,\
                                 InputEncoderv2Decoder,\
                                 InputEncoderAttentionDecoder, Input2EncoderDecoder

import matplotlib.pyplot as plt
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


allys_needed = True
# Main function
# noinspection PyPep8Naming

num_seqs = 5


def training_process():
    last_week_data = Week.fromstring('202106')  # Total weeks: 49
    T = 10
    stride = 1
    total_weeks = 49
    n_min_seqs = 10  # Goes from 5 to 31(totalWeeks - T + stride / stride) - weakAhead
    max_val_week = total_weeks - wk_ahead + 1
    min_val_week = T + n_min_seqs - 1
    # min_val_week = 40
    print("Initializing ...")
    print(device)
    for region in regions:

        print(f'Region: {region}')
        total_data_seq = Dataset(data_path, last_week_data, region, include_col, wk_ahead)
        _, ysT, _, _, allyT = total_data_seq.create_seqs_limited(T, stride, RNN_DIM, get_test=False)
        # _, ysT, _, _, allyT = total_data_seq.create_seqs(n_min_seqs, RNN_DIM)
        ysT = total_data_seq.scale_back_Y(ysT)
        yT = total_data_seq.scale_back_Y(allyT[-1])

        for ew, ew_str in zip(weeks[min_val_week:max_val_week], weeks_strings[min_val_week-1:max_val_week-1]):
            print(f'Week:{ew}')
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
            # seqModel = EncoderDecoder(seqs.shape[-1], RNN_DIM, wk_ahead)  # Change
            # seqModel = InputEncoderDecoder(seqs.shape[1], seqs.shape[-1], RNN_DIM, wk_ahead)
            # seqModel = EncoderAttentionDecoder(RNN_DIM, seqs.shape[-1], RNN_DIM, wk_ahead)
            # seqModel = InputEncoderAttentionDecoder(T, seqs.shape[-1], RNN_DIM, wk_ahead)
            seqModel = InputEncoderDecoderHidden(T, seqs.shape[-1], RNN_DIM, wk_ahead)
            # seqModel = EncoderDecoderHidden(seqs.shape[-1], RNN_DIM, wk_ahead)

            # seqModel = InputEncoderv2Decoder(T, seqs.shape[-1], RNN_DIM, wk_ahead)
            # seqModel = InputEncoderv2Decoder(seqs.shape[1], seqs.shape[-1], RNN_DIM, wk_ahead)

            # seqModel = Input2EncoderDecoder(T, seqs.shape[-1], RNN_DIM, wk_ahead,)
            # Trainig process
            val, loss, test = trainingModel(seqModel, dataset, num_seqs,
                                            0.001, 500, 0.1, 10,
                                            seqs, mask_seq, ys, mask_ys, allys, ysT,
                                            allys_needed=allys_needed,
                                            get_att=False)
            total_epoch1 = len(val)
            twin1.plot(range(total_epoch1), val, c='r')
            twin2.plot(range(total_epoch1), test, c='g')
            ax.plot(range(total_epoch1), loss, c='b')
            # fig.show()
            val = []
            loss = []
            test = []

            val, loss, test = trainingModel(seqModel, dataset, num_seqs,
                                            0.0001, 500, 0.1, 10,
                                            seqs, mask_seq, ys, mask_ys, allys, ysT,
                                            allys_needed=allys_needed,
                                            get_att=False)
            # Ploting loss and eval
            total_epoch2 = len(val) - 1
            twin1.plot(range(total_epoch1-1, total_epoch1 + total_epoch2), val, c='r')
            twin2.plot(range(total_epoch1-1, total_epoch1 + total_epoch2), test, c='g')
            ax.plot(range(total_epoch1-1, total_epoch1 + total_epoch2), loss, c='b')
            val = []
            loss = []
            test = []

            val, loss, test = trainingModel(seqModel, dataset, num_seqs,
                                            0.00001, 500, 0.1, 10,
                                            seqs, mask_seq, ys, mask_ys, allys, ysT,
                                            allys_needed=allys_needed,
                                            get_att=False)
            # Ploting loss and eval
            total_epoch3 = len(val) - 1
            p1, = twin1.plot(range(total_epoch1 + total_epoch2-1, total_epoch1+total_epoch2+total_epoch3),
                             val, c='r', label='Val')
            p2, = twin2.plot(range(total_epoch1 + total_epoch2-1, total_epoch1+total_epoch2+total_epoch3),
                             test, c='g', label='Test')
            p3, = ax.plot(range(total_epoch1 + total_epoch2-1, total_epoch1+total_epoch2+total_epoch3), loss,
                          c='b', label='Loss')
            val = []
            loss = []
            test = []

            ax.set_ylabel("Loss")
            twin1.set_ylabel("Val")
            twin2.set_ylabel("Test")
            ax.yaxis.label.set_color(p3.get_color())
            twin1.yaxis.label.set_color(p1.get_color())
            twin2.yaxis.label.set_color(p2.get_color())
            tkw = dict(size=4, width=1.5)
            ax.tick_params(axis='y', colors=p3.get_color(), **tkw)
            twin1.tick_params(axis='y', colors=p1.get_color(), **tkw)
            twin2.tick_params(axis='y', colors=p2.get_color(), **tkw)
            ax.tick_params(axis='x', **tkw)

            ax.legend(handles=[p1, p2, p3])

            # fig.show()
            # plt.close()
            torch.cuda.empty_cache()

        # Saving the model and plot
            path_model = model_path_save + region + '_' + ew_str + '_' + '.pth'
            torch.save(seqModel.state_dict(), path_model)
            name_image = region + '_' + ew_str + '_' + aproach + '.png'
            fig.savefig(path_figs + name_image)
            ax.clear()
            twin1.clear()
            twin2.clear()
            plt.cla()
            fig.clf()
            gc.collect()
            del seqModel
            seqModel = []
            val = []
            loss = []
            test = []
            seqs = seqs.cpu()
            ys = ys.cpu()
            mask_seq = mask_seq.cpu()
            mask_ys = mask_ys.cpu()
            allys = allys.cpu()

        # plt.show()


# Function to be executed.
if __name__ == '__main__':
    training_process()
