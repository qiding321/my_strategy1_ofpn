# -*- coding: utf-8 -*-
"""
Created on 2016/9/27 14:12

@author: qiding
"""

import os

import pandas as pd

import my_path.path

path_root = my_path.path.market_making_result_root
fold_name = '2016-09-30-16-48-33rolling_fixed_vars_OLS_12M_predict_1M_normalized_by_12M_notdivstd_1min_trancate_method_mean_std_period_20_std_3'

dates = [x for x in os.listdir(path_root + fold_name) if len(x.split('.')) == 1]
predict_date = [x[18:18 + 8] for x in dates]


def _str2datestr_(date_tuple_str):
    t = date_tuple_str.replace('(', '').replace(')', '').replace(' ', '').split(',')
    t2 = [s_.zfill(2) for s_ in t]
    return ''.join(t2)


def get_rsquared(date_):
    data_path = path_root + fold_name + '\\' + date_ + '\\' + 'daily_rsquared.csv'
    data_raw = pd.read_csv(data_path)
    data = data_raw.iloc[2:, :]
    data.columns = ['ymd', 'mse', 'msr', 'rsquared']
    data['ymd'] = data['ymd'].apply(_str2datestr_)
    data.set_index('ymd', inplace=True)
    return data


def get_err(date_):
    data_path = path_root + fold_name + '\\' + date_ + '\\' + 'err_description.csv'
    data_raw = pd.read_csv(data_path, header=None, index_col=0)
    # data = pd.Series(data_raw)
    return data_raw[1]


rsquared_daily_list = list(map(get_rsquared, dates))
rsquared_daily = pd.concat(rsquared_daily_list).sort_index()

err_daily_list = list(map(get_err, dates))
err_daily = pd.concat(err_daily_list, axis=1, keys=predict_date).T

rsquared_daily.to_csv(path_root + fold_name + '\\' + 'rsquared_daily.csv')
err_daily.to_csv(path_root + fold_name + '\\' + 'err_daily.csv')
