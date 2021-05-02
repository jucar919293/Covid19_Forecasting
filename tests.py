from modulesRNN.utils import Dataset
from epiweeks import Week
from seq2seqModels.models import EncoderAttentionDecoder, EncoderDecoderHidden, InputEncodeAttentionDecoder
# General variables
data_path_dataset = './data/train_data_weekly_vEW202105.csv'
data_path_visual = './data/train_data_weekly_noscale_vEW202105.csv'
weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]
weeks = [Week.fromstring(y) for y in weeks_strings]
regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']
include_col = ['target_death', 'retail_and_recreation_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
               'residential_percent_change_from_baseline', 'positiveIncrease', 'negativeIncrease',
               'totalTestResultsIncrease', 'onVentilatorCurrently', 'inIcuCurrently', 'recovered',
               'hospitalizedIncrease', 'death_jhu_incidence', 'dex_a', 'apple_mobility', 'CLI Percent of Total Visits',
               'fb_survey_cli']
wk_ahead = 4

# Things to test:
region = regions[0]
last_week_data = weeks[-1]
dataset = Dataset(data_path_dataset, last_week_data, region, include_col, wk_ahead)
T = 20
stride = 1
RNN_DIM = 128
seqs, ys, mask_seq, mask_ys, allys, test = dataset.create_seqs_limited(T, stride, RNN_DIM, get_test=True)
# seqModel = EncoderDecoderHidden(seqs.shape[-1], RNN_DIM, wk_ahead)
seqModel = InputEncodeAttentionDecoder(T, seqs.shape[-1], RNN_DIM, wk_ahead)
A = seqModel.forward(seqs, mask_seq, allys, ys)
