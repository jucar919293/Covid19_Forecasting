import pandas as pd

# Variables
regions = ['X', 'WA', 'TX', 'PA', 'LA']
model_types = ('2ED', 'ED', 'Input2ED', 'InputED')
cyc = '_0'

error_path = './Results/Errors/' + model_types[1] + '_' + regions[0] + cyc + '.csv'
error = pd.read_csv(error_path)