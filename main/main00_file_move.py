
import os
import shutil

import pandas as pd

path0 = r'C:\Users\dqi\Documents\Output\MarketMaking\\'
path1 = r'C:\Users\dqi\Documents\Output\MarketMaking\\different_time_scale\\'

name_ = 'result_best.csv'

file_list = [p for p in os.listdir(path0) if p.startswith('2016-09-25')]

data_list = []

for fold in file_list:
    file_out = path0 + fold + '\\' + name_
    # file_in = path1 + fold + '.csv'
    # shutil.copy(file_out, file_in)
    data = pd.read_csv(file_out)[['time_period', 'rsquared_oos']]
    data = data.rename(columns={'rsquared_oos': fold})
    data['time_oos'] = data['time_period'].apply(lambda x: x[-8:])
    data.set_index('time_oos', inplace=True)
    data.drop('time_period', axis=1, inplace=True)
    data_list.append(data)

data_df = pd.concat(data_list, axis=1)
file_in = path1 + 'agg.csv'
data_df.to_csv(file_in)