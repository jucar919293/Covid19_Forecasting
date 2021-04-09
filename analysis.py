import pandas as pd
import matplotlib.pyplot as plt

# Variables
regions = ['X', 'CA', 'FL', 'GA', 'IL', 'LA', 'PA', 'TX', 'WA']

model_types = ('2ED', 'ED', 'Input2ED', 'InputED')
cyc = '_0'
model_type = model_types[2]
region = regions[0]

predictions_path = './Results/Predictions/' + model_type + '_' + region + cyc + '.csv'
predictions = pd.read_csv(predictions_path)

real_path = './Results/Real/' + model_type + '_' + region + cyc + '.csv'
real = pd.read_csv(real_path)
weeks = ['wk1', 'wk2', 'wk3', 'wk4']
for i in range(1):
    plt.plot(range(i, 26+i), predictions[weeks[i]], 'o')
plt.legend(['1 week ahead', 'Observations'])

plt.plot(range(0, 26), real['wk1'], '-')

plt.show()
