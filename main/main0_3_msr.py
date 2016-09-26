import os

import pandas as pd


def main():
    fold_path = r'C:\Users\dqi\Documents\Output\MarketMaking\2016-09-25-17-50-52rolling_model_selection_12M_predict_1M_normalized_by_12M_10min_10min\\'
    output_file_name = 'msr_all.csv'

    date_list = os.listdir(fold_path + 'detail\\')

    time_period_list = []
    time_oos = []
    model_len = []
    var_names = []
    msr_list = []

    for date in date_list:
        print(date)
        var_num_list = os.listdir(fold_path + 'detail\\' + date + '\\')
        for var_num in var_num_list:
            reg_num_list = os.listdir(fold_path + 'detail\\' + date + '\\' + var_num + '\\')
            for reg_num in reg_num_list:
                try:
                    file_path = fold_path + 'detail\\' + date + '\\' + var_num + '\\' + reg_num + '\\result.csv'
                    info = get_info_from_csv(file_path)
                    msr = info['msr']
                    var_name = info['var_names']

                    time_period_list.append(date)
                    time_oos.append(date[-8:])
                    model_len.append(var_num[8:])
                    var_names.append(var_name)
                    msr_list.append(msr)
                except Exception as e:
                    file_path = fold_path + 'detail\\' + date + '\\' + reg_num + '\\result.csv'
                    print(e)
                    print(file_path)
    df = pd.DataFrame(
        [time_period_list, time_oos, model_len, var_names, msr_list],
        index=['time_period', 'time_oos', 'model_len', 'var_names', 'msr_list']
    ).T
    df.to_csv(fold_path + output_file_name, index=None)


def get_info_from_csv(file_path):
    info = {}
    with open(file_path, 'r') as f_in:
        for line_count, line in enumerate(f_in.readlines()):
            if line_count == 1:
                info['var_names'] = line_split(line)
            if line_count == 2:
                info['coef'] = line_split(line)
            if line_count == 3:
                info['tvalues'] = line_split(line)
            if line_count == 5:
                info['rsquared_insample'] = line_split(line)[0]
            if line_count == 8:
                info['mse'] = line_split(line)[0]
            if line_count == 9:
                info['msr'] = line_split(line)[0]
            if line_count == 10:
                info['rsquared_oos'] = line_split(line)[0]
            if line_count == 11:
                info['rsquared_by_oos_mean'] = line_split(line)[0]
    return info


def line_split(line):
    lst = line.replace('\n', '').split(',')
    return lst[1:]


if __name__ == '__main__':
    main()
