import os

import pandas as pd

import my_path.path


def main():
    root_path = my_path.path.market_making_result_root
    fold_name_list = [
        fold_name_ for fold_name_ in os.listdir(root_path) if fold_name_.startswith('2016-10-05')
        ]

    df_list = []
    name_list = []

    for fold_name in fold_name_list:
        name_list.append(fold_name[-24:])
        r_sq_oos = pd.read_csv(root_path + fold_name + '\\r_squared_record.csv', index_col=None)
        r_sq_oos_ = r_sq_oos[['time_period_out_of_sample', 'rsquared_out_of_sample']]
        r_sq_oos_.set_index('time_period_out_of_sample', inplace=True)
        r_sq_oos_['buyvolume_trancated_ratio'] = pd.np.nan
        tpo = [time_period_oos for time_period_oos in os.listdir(root_path + fold_name) if time_period_oos.startswith('20')]
        ratio_list = []
        for tpo_ in tpo:
            length_data_ = pd.read_csv(root_path + fold_name + '\\' + tpo_ + '\\' + 'len_record_predicting.csv', header=None)
            length_data_.set_index(0, inplace=True)
            buy_trancated_ratio = length_data_[1]['buyvolume_trancated'] / length_data_[1]['final_data_len']
            ratio_list.append(buy_trancated_ratio)
            r_sq_oos_['buyvolume_trancated_ratio'][tpo_] = buy_trancated_ratio

        df_this_fold = r_sq_oos_
        df_list.append(df_this_fold)
    df = pd.concat(df_list, axis=1, keys=name_list)
    df.to_csv(root_path + 'trancated_r_sq_and_length_agg.csv')


if __name__ == '__main__':
    main()
