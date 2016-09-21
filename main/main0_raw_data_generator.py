# -*- coding: utf-8 -*-
"""
Created on 2016/9/13 14:58

@author: qiding
"""

# -*- coding: utf-8 -*-

import os
import pandas as pd
import datetime
import copy

import util.util as util
from log.log import log_order_flow_predict
from log.log import log_error
import paras.resample_dicts as resample_dicts
import util.const as const


def main():
    output_folder = 'I:\\OrderFlowPredictData_from_raw_data\\601818\\'

    stk_input_folder = 'I:\\601818_and_index_order_data_raw\\Tick\\SH\\'
    trans_input_folder = 'I:\\601818_and_index_order_data_raw\\Transaction\\SH\\'
    index_input_folder = 'I:\\601818_and_index_order_data_raw\\Tick\\SH\\'
    index_future_input_folder = 'F:\\IF_complete_data\\'
    # stk_input_folder = 'I:\\IntradayDataOwT\\SH\\'
    # index_input_folder = 'F:\\IntradayIndex\\Tick\\SH\\'

    start_time = '20130101'

    input_output_file_mapping_list = generate_file_mapping_list(stk_input_folder, index_input_folder, trans_input_folder, index_future_input_folder, output_folder, start_time)  # [(index_in, stk_in, transaction_in, index_future_in, file_out)]
    log_order_flow_predict.info('file_mapping_list: %d' % len(input_output_file_mapping_list))

    for index_in, stk_in, transaction_in, index_future_in, file_out in input_output_file_mapping_list:

        log_order_flow_predict.info(file_out + ' begin')

        index_data_50 = pd.read_csv(index_in[0], parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'), encoding='gbk')[['date', 'time', 'price', 'volume', 'accvolume']].drop_duplicates('time', keep='last').set_index('time')
        index_data_300 = pd.read_csv(index_in[1], parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'), encoding='gbk')[['date', 'time', 'price', 'volume', 'accvolume']].drop_duplicates('time', keep='last').set_index('time')
        stk_data = pd.read_csv(stk_in, parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'), encoding='gbk').drop_duplicates('time', keep='last').set_index('time')
        transaction_data = pd.read_csv(transaction_in, parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'), encoding='gbk').set_index('time')
        index_future_data = _get_index_future_data(index_future_in)

        index_data_resample1 = clean_data_index(index_data_50, name='sh50')
        index_data_resample2 = clean_data_index(index_data_300, name='hs300')
        stk_data_resample = clean_data_stk(stk_data)
        transaction_data_resample = clean_data_transaction(transaction_data)
        index_future_data_resample = clean_data_index_future(index_future_data, name='index_future_300')

        data_merged = pd.DataFrame(pd.concat([
            index_data_resample1,
            index_data_resample2,
            stk_data_resample,
            transaction_data_resample,
            index_future_data_resample
        ], axis=1))

        data_merged['date_index'] = data_merged['date_index_sh50'].fillna(method='ffill').fillna(method='bfill')
        data_merged['time'] = list(map(lambda time_: _date_time_merge(time_[0], time_[1]), zip(data_merged.index, data_merged['date_index'])))
        col = sorted(data_merged.columns, key=lambda x: x[::-1])
        col.remove('time')
        col.insert(0, 'time')
        data_merged_sorted_columns = data_merged.reindex(columns=col)
        data_merged_sorted_columns.to_csv(file_out, index=None)
        log_order_flow_predict.info(file_out + ' end')


def _get_index_future_data(index_future_path):
    data = pd.read_csv(index_future_path, encoding='gbk', date_parser=lambda x: datetime.datetime.strptime(str(x)[11:], '%H:%M:%S.%f'), parse_dates=['时间'])  # todo
    data_new_name = data.rename(columns={'时间': 'time', '最新': 'price', '成交量': 'volume'})
    data_new_name['accvolume'] = data_new_name['volume'].cumsum()

    data_to_ret = data_new_name[['time', 'price', 'volume', 'accvolume']].drop_duplicates('time', keep='last').set_index('time')
    return data_to_ret


def _date_time_merge(time_, date_):
    year = int(str(date_)[0:4])
    month = int(str(date_)[4:6])
    day = int(str(date_)[6:8])
    hour = time_.hour
    minute = time_.minute
    second = time_.second
    time = datetime.datetime(year, month, day, hour, minute, second)
    return time


def _get_index_future_path(path_root, date):
    path0 = path_root + 'sf2014\\' + date + '\\'
    if os.path.exists(path0):
        pass
    else:
        path0 = path_root + 'sf2013\\' + date + '\\'
    if os.path.exists(path0):
        pass
    else:
        return None
    file_name_list = os.listdir(path0)
    file_name_list_if = sorted([f_n_ for f_n_ in file_name_list if f_n_[0:2] == 'IF' and f_n_[-3:] == 'csv'])
    file_name = file_name_list_if[0]
    return path0 + file_name


def generate_file_mapping_list(stk_input_folder, index_input_folder, trans_input_folder, index_future_input_folder, output_folder, start_time=None):
    dates1 = os.listdir(index_input_folder)
    dates2 = os.listdir(stk_input_folder)
    dates = [d_ for d_ in dates1 if d_ in dates2 and (d_ >= start_time if start_time is not None else True)]
    mapping_list = []
    for date in dates:
        index_in1 = index_input_folder + date + '\\' + '999987.csv'
        index_in2 = index_input_folder + date + '\\' + '000300.csv'
        stk_in = stk_input_folder + date + '\\' + '601818.csv'
        transaction_in = trans_input_folder + date + '\\' + '601818.csv'
        file_out = output_folder + date + '.csv'
        index_future_in = _get_index_future_path(index_future_input_folder, date)
        if index_future_in is None:
            log_error.info('file not exist: %s' % index_future_input_folder + date)
        elif all(os.path.exists(path_) for path_ in [index_in1, index_in2, stk_in, transaction_in, index_future_in]):
            mapping_list.append(((index_in1, index_in2), stk_in, transaction_in, index_future_in, file_out))
        else:
            [
                log_error.info('file not exist: %s' % file_path_)
                for file_path_ in [index_in1, index_in2, stk_in, transaction_in, index_future_in]
                if not os.path.exists(file_path_)
                ]

    return mapping_list


def clean_data_index(index_data, freq='3s', name=''):

    index_data_filtered = filter_time(index_data)

    fun_dict = {
        'date': 'last',
        'price': 'last',
        'volume': 'sum',
        'accvolume': 'last'
    }
    rename_dict = {
        'date': 'date_index'+'_'+name,
        'price': 'price_index'+'_'+name,
        'volume': 'volume_index'+'_'+name,
        'accvolume': 'accvolume_index'+'_'+name
    }

    resample_data = index_data_filtered.resample(freq, label='right', closed='left').apply(fun_dict).rename(columns=rename_dict)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_data_index_future(index_data, freq='3s', name=''):

    index_data_filtered = filter_time(index_data)

    fun_dict = {
        'price': 'last',
        'volume': 'sum',
        'accvolume': 'last'
    }
    rename_dict = {
        'price': 'price_index'+'_'+name,
        'volume': 'volume_index'+'_'+name,
        'accvolume': 'accvolume_index'+'_'+name
    }

    resample_data = index_data_filtered.resample(freq, label='right', closed='left').apply(fun_dict).rename(columns=rename_dict)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_data_stk(stk_data, freq='3s'):
    func_dict = resample_dicts.stk_func_dict

    cols = list(set(stk_data.columns).intersection(set(func_dict.keys())))
    func_dict2 = dict((x_, func_dict[x_]) for x_ in cols)

    resample_data = stk_data.resample(freq, label='right', closed='left')[cols].agg(func_dict2)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_data_transaction(transaction_data, freq='3s'):
    transaction_data['amount'] = transaction_data['trade_price'] * transaction_data['trade_volume']

    transaction_data_buy = transaction_data[transaction_data['bs_flag'] == ord('B')]
    transaction_data_sell = transaction_data[transaction_data['bs_flag'] == ord('S')]

    columns = [
        'newprice',
        'totalamount', 'totalvolume', 'totaltransaction',
        'buytrans', 'selltrans',
        'buyvolume', 'sellvolume',
        'buyamount', 'sellamount'
        ]

    # def func_for_trans(chunk_new):
    #     # chunk_new = copy.deepcopy(chunk)
    #     if len(chunk_new) == 0:
    #         return None
    #     buy_chunk = chunk_new[chunk_new['bs_flag'] == ord('B')]
    #     sell_chunk = chunk_new[chunk_new['bs_flag'] == ord('S')]
    #
    #     newprice = chunk_new['trade_price'].iloc[-1]
    #     totalamount = chunk_new['amount'].sum()
    #     totalvolume = chunk_new['trade_price'].sum()
    #     totaltransaction = chunk_new['trade_price'].count()
    #
    #     buytrans = buy_chunk['trade_price'].count()
    #     buyvolume = buy_chunk['trade_volume'].sum()
    #     buyamount = buy_chunk['amount'].sum()
    #
    #     selltrans = sell_chunk['trade_price'].count()
    #     sellvolume = sell_chunk['trade_volume'].sum()
    #     sellamount = sell_chunk['amount'].sum()
    #
    #     s = pd.Series([newprice,
    #                    totalamount, totalvolume, totaltransaction,
    #                    buytrans, selltrans,
    #                    buyvolume, sellvolume,
    #                    buyamount, sellamount
    #                    ], index=columns)
    #     return s

    def resample_sum(s_):
        return s_.resample(freq, label='right', closed='left').sum()

    def resample_last(s_):
        return s_.resample(freq, label='right', closed='left').apply('last')

    def resample_count(s_):
        return s_.resample(freq, label='right', closed='left').count()

    newprice = resample_last(transaction_data['trade_price'])
    totalamount = resample_sum(transaction_data['amount'])
    totalvolume = resample_sum(transaction_data['trade_volume'])
    totaltransaction = resample_count(transaction_data['trade_volume'])

    buyamount = resample_sum(transaction_data_buy['amount'])
    buyvolume = resample_sum(transaction_data_buy['trade_volume'])
    buytrans = resample_count(transaction_data_buy['trade_volume'])

    sellamount = resample_sum(transaction_data_sell['amount'])
    sellvolume = resample_sum(transaction_data_sell['trade_volume'])
    selltrans = resample_count(transaction_data_sell['trade_volume'])

    resample_data = pd.DataFrame([
        newprice,
        totalamount, totalvolume, totaltransaction,
        buytrans, selltrans,
        buyvolume, sellvolume,
        buyamount, sellamount
        ], index=columns).T

    # resample_data0 = transaction_data.resample(freq).agg(func_for_trans)
    # resample_data = resample_data0.unstack()

    # resample_index = pd.Series(transaction_data.resample(freq, label='right', closed='left').index)
    #
    # def mapping_func(time_):
    #     return len(resample_index[resample_index <= time_])
    # resample_index.apply(mapping_func)
    #
    # resample_data0 = transaction_data.groupby(mapping_func).apply(func_for_trans)
    # rename_dict = dict(map(lambda ri_: (mapping_func(ri_), ri_), resample_index.values))
    # resample_data = resample_data0.rename(index=rename_dict)
    #
    resample_data2 = filter_time(resample_data)

    return resample_data2


def filter_time(data):
    data_new = data.select(lambda x: const.MARKET_OPEN_TIME <= x <=const.MARKET_CLOSE_TIME_NOON or const.MARKET_OPEN_TIME_NOON<= x <= const.MARKET_END_TIME)
    return data_new


if __name__ == '__main__':
    main()
