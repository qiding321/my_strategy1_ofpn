# -*- coding: utf-8 -*-
"""
Created on 2016/8/30 10:34

@author: qiding
"""

import datetime
import os

import pandas as pd

import log.log
import util.const as const


def str2date_ymdhms(date_str):
    date_datetime = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return date_datetime


def get_windows(time_scale_long, time_scale_short='3s'):
    td0 = pd.datetools.to_offset(time_scale_short).delta
    td1 = pd.datetools.to_offset(time_scale_long).delta
    windows = int(td1 / td0)
    return windows


def get_seconds(start_time=const.MARKET_OPEN_TIME, end_time=const.MARKET_END_TIME):
    seconds = (end_time - start_time).total_seconds()
    if end_time >= const.MARKET_OPEN_TIME_NOON and start_time <= const.MARKET_CLOSE_TIME_NOON:
        seconds -= 1.5 * 3600
    return seconds


def is_in_market_open_time(time):
    time_new = datetime.datetime(1900, 1, 1, time.hour, time.minute, time.second)
    b = const.MARKET_OPEN_TIME <= time_new <= const.MARKET_CLOSE_TIME_NOON or const.MARKET_OPEN_TIME_NOON <= time_new <= const.MARKET_END_TIME
    return b


################# too time consuming
# def in_intraday_period(time, time_period='10min'):
#     time_ = datetime.datetime(1900, 1, 1, hour=time.hour, minute=time.minute, second=time.second)
#     seconds = get_seconds(end_time=time_)
#     td0 = datetime.timedelta(seconds=seconds)
#     td1 = pd.datetools.to_offset(time_period).delta
#     try:
#         period = int(td0 / td1)
#     except ZeroDivisionError:
#         period = 0
#     return period

def in_intraday_period(time, time_period='10min'):
    time_ = datetime.datetime(1900, 1, 1, hour=time.hour, minute=time.minute, second=time.second)
    seconds = get_seconds(end_time=time_)
    # td0 = datetime.timedelta(seconds=seconds)
    # td1 = datetime.timedelta(seconds=600)
    if time_period == '10min':
        time_period_seconds = 600
    else:
        log.log.log_price_predict.error('time_period_error: {}'.format(time_period))
        raise ValueError
    period = int(seconds / time_period_seconds)
    return period


def datetime2ymdstr(time):
    s = time.strftime('%Y-%m-%d')
    return s


def my_resample(my_data, time_freq):
    new_data = my_data.groupby(datetime2ymdstr, group_keys=False).apply(lambda x: x.resample(time_freq).apply('mean').select(is_in_market_open_time))
    return new_data


def drop_na_for_x_and_y(x, y):
    d = pd.DataFrame(pd.concat([x, y], axis=1, keys=['x', 'y']))
    d2 = d.dropna()
    x2 = d2['x']
    y2 = d2['y']
    # print('len_x: {}\nlen_y: {}\nlen_x_new: {}\nlen_y_new: {}'.format(len(x), len(y), len(x2), len(y2)))

    return x2, y2


def pandas2str(df, title=''):
    s = ''
    s += title + '\n'
    s += ','
    s += ','.join([str(s_) for s_ in df.columns])
    s += '\n'
    s += '\n'.join([str(k) + ',' + ','.join([str(v_) for v_ in v.values]) for k, v in df.iterrows()])
    s += '\n\n'

    return s


def dict2str(d):
    s = '\n'.join(['{}: {}'.format(k, v) for k,v in d.items()])
    return s


def get_timenow_str():
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d-%H-%M-%S')
    return now_str


def high_order_name_change(name_list, order=2):
    r = [x_var_ + '_order2' for x_var_ in name_list]
    return r


def record_result(to_record_str, to_record_path):
    if os.path.exists(to_record_path):
        pass
    else:
        os.makedirs(to_record_path)
    with open(to_record_path + 'result.csv', 'a') as f_out:
        f_out.write(to_record_str)
