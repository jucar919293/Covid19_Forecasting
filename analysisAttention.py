import pandas as pd
import matplotlib.pyplot as plt
from modulesRNN.utils import Dataset
from epiweeks import Week

# Variables
regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']

model_types = ('2ED', 'ED', 'Input2ED', 'InputED')
cyc = '_0'
model_type = model_types[2]
region = regions[1]
include_col = ['target_death', 'retail_and_recreation_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
               'residential_percent_change_from_baseline', 'positiveIncrease', 'negativeIncrease',
               'totalTestResultsIncrease', 'onVentilatorCurrently', 'inIcuCurrently', 'recovered',
               'hospitalizedIncrease', 'death_jhu_incidence', 'dex_a', 'apple_mobility', 'CLI Percent of Total Visits',
               'fb_survey_cli']

data_path = './data/train_data_weekly_vEW202105.csv'
last_week_data = Week.fromstring('202106')
wk_ahead = 4

weeks_strings = [str(x) for x in range(202010, 202054)] + [str(y) for y in range(202101, 202107)]
weeks = [Week.fromstring(y) for y in weeks_strings]

total_data_seq = Dataset(data_path, last_week_data, region, include_col, wk_ahead)


for i in range(20, 36):
    fig, ax1 = plt.subplots()
    att_path = './Results/Attention/at_' + model_type + '_' + region + str(i) + cyc + '.csv'
    print(att_path)
    attention = pd.read_csv(att_path).transpose()
    ax1.plot(total_data_seq.y[:i], ':')
    ax2 = ax1.twinx()
    ax2.plot(attention)
    fig.tight_layout()

    plt.legend(include_col)
    plt.show()
