import torch
from epiweeks import Week
import matplotlib.pyplot as plt
from modulesRNN.utils import Dataset
from seq2seqModels import encoder_decoder, two_encoder_decoder, inputAttention2ED, inputAttentionED
import modulesRNN.utils as utils
import pandas as pd
from numpy import array
device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# Introduce the path were the data is storage
data_path_dataset = './data/train_data_weekly_vEW202105.csv'
data_path_visual = './data/train_data_weekly_noscale_vEW202105.csv'
wk_ahead = 4
# regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']
regions = ['X']
weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]
weeks = [Week.fromstring(y) for y in weeks_strings]
include_col = ['target_death', 'retail_and_recreation_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
               'residential_percent_change_from_baseline', 'positiveIncrease', 'negativeIncrease',
               'totalTestResultsIncrease', 'onVentilatorCurrently', 'inIcuCurrently', 'recovered',
               'hospitalizedIncrease', 'death_jhu_incidence', 'dex_a', 'apple_mobility', 'CLI Percent of Total Visits',
               'fb_survey_cli']
RNN_DIM = 128  # Change
n_signals = len(include_col) - 1
last_week_data = Week.fromstring('202106')

model_types = ('ED', '2ED', 'Input2ED', 'InputED')


# noinspection DuplicatedCode
def testing():
    # Variables:
    n_week = 45
    wk_ahead = 4
    model_type = model_types[2]  # Change
    error_total = {}
    predictions_total = {}
    real_values_total = {}

    mae = []
    mse = []
    mape = []
    rmse = []
    cyc = '_0'
    error_region = {}
    for region in regions:
        for n_week in range(20, 36):
            # for n_week in range(20, 49 - wk_ahead + 1):

            # region = 'X'
            week = weeks[n_week]  # Actual week +1
            path_model = './trainedModels/firstTry/' + model_type + '/' + region + '_' \
                         + weeks_strings[n_week - 1] + cyc + '.pth'
            print(path_model)
            dataset_visual = Dataset(data_path_visual, last_week_data, region, include_col, wk_ahead)

            # Creating and loading model
            dataset_test = Dataset(data_path_dataset, week, region, include_col, wk_ahead)
            seqs, ys, mask_seq, mask_ys, allys = dataset_test.create_seqs(dataset_test.y.shape[0], RNN_DIM)
            seq_model = inputAttention2ED.Seq2SeqModel(seqs.shape[1], seqs.shape[-1], RNN_DIM, wk_ahead)  # Change
            seq_model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
            # Testing the model
            seq_model.eval()
            # Allys is for two encoder models
            predictions_no_scaled, df_e, df_at = seq_model(seqs, mask_seq, allys, get_att=True)  # Change in att.
            # predictions_no_scaled = seq_model(seqs, mask_seq, allys)  # Change
            predictions_tensor = dataset_test.scale_back_Y(predictions_no_scaled)
            predictions = predictions_tensor.detach().numpy()
            # Obtaining real values
            real_values = dataset_visual.y[n_week:n_week + wk_ahead]
            real_values_tensor = torch.tensor(real_values)
            # Calculating error
            mae.append(array([utils.mae_calc(real_values_tensor[i],
                       predictions_tensor[i]).detach().numpy() for i in range(wk_ahead)]))
            mape.append(array([utils.mape_calc(real_values_tensor[i],
                        predictions_tensor[i]).detach().numpy() for i in range(wk_ahead)]))
            mse.append(array([utils.mse_calc(real_values_tensor[i],
                       predictions_tensor[i]).detach().numpy() for i in range(wk_ahead)]))
            rmse.append(array([utils.rmse_calc(real_values_tensor[i],
                        predictions_tensor[i]).detach().numpy() for i in range(wk_ahead)]))

            predictions_total[weeks_strings[n_week - 1]] = predictions
            real_values_total[weeks_strings[n_week - 1]] = real_values

            # Plotting an printing results
            x = [n for n in range(n_week+1, n_week + wk_ahead+1)]
            plt.plot(x, real_values, '-')
            plt.plot(x, predictions, 'o')

            # Change in att.
            # df_e.to_csv('./Results/Attention/e_' + model_type + '_' + region + str(n_week) + cyc + '.csv',
            #             index=False, header=True)
            # df_at.to_csv('./Results/Attention/at_' + model_type + '_' + region + str(n_week) + cyc + '.csv',
            #              index=False, header=True)
            print(f'N:{n_week} MAE:{mae}, MAPE:{mape}, MSE:{mse}, RMSE:{rmse}')

        error_region['mae'] = array(mae).mean(0)
        error_region['mape'] = array(mape).mean(0)
        error_region['mse'] = array(mse).mean(0)
        error_region['rmse'] = array(rmse).mean(0)

        df_e = pd.DataFrame(error_region, index=['wk1', 'wk2', 'wk3', 'wk4']).transpose()
        df_p = pd.DataFrame(predictions_total, index=['wk1', 'wk2', 'wk3', 'wk4']).transpose()
        df_r = pd.DataFrame(real_values_total, index=['wk1', 'wk2', 'wk3', 'wk4']).transpose()

        df_e.to_csv('./Results/Errors/'+model_type+'_'+region+cyc+'.csv', index=False, header=True)
        df_p.to_csv('./Results/Predictions/'+model_type+'_'+region+cyc+'.csv', index=False, header=True)
        df_r.to_csv('./Results/Real/'+model_type+'_'+region+cyc+'.csv', index=False, header=True)

        plt.show()


if __name__ == '__main__':
    testing()
