# -*- coding: utf-8 -*-
"""
Created on 2016/8/30 10:10

@author: qiding
"""

import datetime
import os
import re

import numpy as np
import pandas as pd

import data.reg_data
import log.log
import my_path.path
import util.util

my_log = log.log.log_order_flow_predict


class DataBase:
    def __init__(self, date_begin='', date_end='', para_dict=None, have_data_df=False, data_df=None, x_names=None, y_names=None):
        assert isinstance(date_begin, str) and isinstance(date_end, str)
        assert isinstance(para_dict, dict)
        self.source_data_path = my_path.path.data_source_path
        self.para_dict = para_dict
        if not have_data_df:
            self.date_begin = date_begin
            self.date_end = date_end

            if 'x_vars_moving_average' in self.para_dict.keys():
                self.data_df = self._get_data_add_20_day_before(data_path=self.source_data_path, date_begin=date_begin,
                                                                date_end=date_end)
                for var_moving_average in self.para_dict['x_vars_moving_average']:
                    self.data_df[var_moving_average] = self._get_one_col(var_moving_average)
            else:
                self.data_df = self._get_data(data_path=self.source_data_path, date_begin=date_begin, date_end=date_end)
                # if 'buy_vol_10min_intraday_pattern_20_days' in self.para_dict['x_vars'] or 'sell_vol_10min_intraday_pattern_20_days' in self.para_dict['x_vars']:
            #     self.data_df = self._get_data_add_20_day_before(data_path=self.source_data_path, date_begin=date_begin, date_end=date_end)
            #
            #     # pre-treat intraday_pattern
                #     for col in ['buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days']:
            #         if col in self.para_dict['x_vars']:
            #             self.data_df[col] = self._get_one_col(col)
            # else:
            #     self.data_df = self._get_data(data_path=self.source_data_path, date_begin=date_begin, date_end=date_end)
        else:
            data_df = data_df.sort_index()
            self.date_begin = data_df.index[0]
            self.date_end = data_df.index[-1]
            self.data_df = data_df
            # self.x_names = x_names
            # self.x_names = (
            #     self.para_dict['x_vars'] +
            #     (self.para_dict['x_vars_moving_average'] if 'x_vars_moving_average' in self.para_dict.keys() else []) +
            #     (util.util.high_order_name_change(self.para_dict['high_order2_term_x']) if 'high_order2_term_x' in self.para_dict.keys() else [])
            # ) if x_names is None else x_names
            # self.y_names = y_names

    def generate_reg_data(self, normalize_funcs=None, normalize=True):

        time_scale_x = self.para_dict['time_scale_x']
        time_scale_y = self.para_dict['time_scale_y']
        time_scale_now = self.para_dict['time_freq']

        key = 'x_{}_y_{}'.format(time_scale_x, time_scale_y)
        my_log.info(key + ' start')

        # generate series to reg
        y_series, x_series = self._get_series_to_reg()

        # lag and normalize
        if normalize_funcs is None:
            x_series_new, y_series_new, normalize_funcs = self._get_useful_lag_series(x_series, y_series, time_scale_x, time_scale_y, time_scale_now, normalize=normalize)
            x_series_rename = x_series_new.rename(columns=dict([(name, name+'_x') for name in x_series_new]))
            y_series_rename = y_series_new.rename(columns=dict([(name, name+'_y') for name in y_series_new]))

            reg_data = data.reg_data.RegDataTraining(x_vars=x_series_rename, y_vars=y_series_rename)

        else:
            x_series_new, y_series_new, normalize_funcs = self._get_useful_lag_series(x_series, y_series, time_scale_x, time_scale_y, time_scale_now, type_='predict', predict_funcs=normalize_funcs, normalize=normalize)
            x_series_rename = x_series_new.rename(columns=dict([(name, name+'_x') for name in x_series_new]))
            y_series_rename = y_series_new.rename(columns=dict([(name, name+'_y') for name in y_series_new]))

            reg_data = data.reg_data.RegDataTest(x_vars=x_series_rename, y_vars=y_series_rename)

        return reg_data, normalize_funcs

    @classmethod
    def _get_data(cls, data_path, date_begin, date_end):
        file_list = os.listdir(data_path)
        date_list = [x.split('.')[0] for x in file_list]
        date_list_useful = [date_ for date_ in date_list if date_begin <= date_ <= date_end]
        path_list_useful = [data_path + date_ + '.csv' for date_ in date_list_useful]
        data_list = [pd.read_csv(path_, date_parser=util.util.str2date_ymdhms, parse_dates=['time']) for path_ in path_list_useful]
        data_df = pd.DataFrame(pd.concat(data_list, ignore_index=True)).set_index('time').sort_index()
        return data_df

    @classmethod
    def _get_data_add_20_day_before(cls, data_path, date_begin, date_end):
        file_list = os.listdir(data_path)
        date_list = sorted([x.split('.')[0] for x in file_list])

        date_begin_actual = [date_ for date_ in date_list if date_begin <= date_ <= date_end][0]
        date_begin_idx = date_list.index(date_begin_actual)
        date_begin_20day_before_idx = max(0, date_begin_idx - 20)
        date_begin_20day_before = date_list[date_begin_20day_before_idx]

        date_list_useful = [date_ for date_ in date_list if date_begin_20day_before <= date_ <= date_end]
        path_list_useful = [data_path + date_ + '.csv' for date_ in date_list_useful]
        data_list = [pd.read_csv(path_, date_parser=util.util.str2date_ymdhms, parse_dates=['time']) for path_ in path_list_useful]
        data_df = pd.DataFrame(pd.concat(data_list, ignore_index=True)).set_index('time').sort_index()

        # drop up-limit or down-limit cases
        data_df[(data_df['bid1'] == 0) | (data_df['ask1'] == 0)] = np.nan

        return data_df

    def _get_useful_lag_series(self, x_vars, y_vars, time_scale_x, time_scale_y, time_scale_now, type_='training', predict_funcs=None, normalize=True):  # todo  check
        assert type_ in ['training', 'predict']

        window_x = util.util.get_windows(time_scale_long=time_scale_x, time_scale_short=time_scale_now)
        window_y = util.util.get_windows(time_scale_long=time_scale_y, time_scale_short=time_scale_now)
        assert window_y >= window_x

        if 'x_vars_moving_average' in self.para_dict.keys():
            contemporaneous_cols = self.para_dict['x_vars_moving_average']
        else:
            contemporaneous_cols = [col_ for col_ in ['buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days'] if col_ in x_vars.columns]
        non_contemp_cols = [col_ for col_ in x_vars.columns if col_ not in contemporaneous_cols]

        x_vars_not_contemp = x_vars[non_contemp_cols]
        x_vars_contemp = x_vars[contemporaneous_cols]

        x_series_not_contemp = x_vars_not_contemp.groupby(util.util.datetime2ymdstr, group_keys=False).apply(
            lambda x: x.resample(time_scale_x, label='right').mean().select(util.util.is_in_market_open_time)
        )

        # add lag term
        if 'lag_terms' in self.para_dict.keys():
            for col_, lag_list in self.para_dict['lag_terms']:
                for lag_ in lag_list:
                    if lag_ == 1:
                        continue
                    name_ = col_+'lag'+str(lag_)
                    x_series_not_contemp[name_] = x_series_not_contemp[col_].shift(lag_-1)
        else:
            for col_ in ['buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change', 'mid_px_ret_dummy', 'mid_px_ret', 'ret_hs300', 'ret_index_index_future_300', 'ret_sh50']:
                if col_ in x_series_not_contemp.columns:
                    for lag_ in [2, 3, 4, 5, 6]:
                        name_ = col_+'lag'+str(lag_)
                        x_series_not_contemp[name_] = x_series_not_contemp[col_].shift(lag_-1)

        # for contemporaneous x-vars
        x_series_contemp = x_vars_contemp.groupby(util.util.datetime2ymdstr, group_keys=False).apply(
            lambda x:
            x.rolling(window=window_y).mean().shift(-window_y)
                .resample(time_scale_x, label='right').apply('last')
                .select(util.util.is_in_market_open_time)
        )
        x_series = pd.merge(x_series_contemp, x_series_not_contemp, left_index=True, right_index=True, how='outer')

        # y series
        y_series = y_vars.groupby(util.util.datetime2ymdstr, group_keys=False).apply(
            lambda x:
            x.rolling(window=window_y).mean().shift(-window_y)
                .resample(time_scale_x, label='right').apply('last')
                .select(util.util.is_in_market_open_time)
        )

        x_series_drop_na, y_series_drop_na = util.util.drop_na_for_x_and_y(x_series, y_series)

        if type_ == 'training':
            if normalize:
                x_series_normalize_func = lambda x_: pd.DataFrame([(x_[col] - x_series_drop_na[col].mean())/x_series_drop_na[col].std() if col != 'mid_px_ret_dummy' else x_[col] for col in x_]).T
                y_series_normalize_func = lambda y_: pd.DataFrame([(y_[col] - y_series_drop_na[col].mean())/y_series_drop_na[col].std() if col != 'mid_px_ret_dummy' else y_[col] for col in y_]).T
            else:
                x_series_normalize_func, y_series_normalize_func = lambda x: x, lambda x: x
        else:
            if normalize:
                x_series_normalize_func = predict_funcs['x_series_normalize_func']
                y_series_normalize_func = predict_funcs['y_series_normalize_func']
            else:
                x_series_normalize_func, y_series_normalize_func = lambda x: x, lambda x: x

        x_new = x_series_normalize_func(x_series_drop_na)
        y_new = y_series_normalize_func(y_series_drop_na)
        predict_funcs = {'x_series_normalize_func': x_series_normalize_func, 'y_series_normalize_func': y_series_normalize_func}
        return x_new, y_new, predict_funcs

    def _get_series_to_reg(self):

        time_freq = self.para_dict['time_freq']

        y_vars_name = self.para_dict['y_vars']
        y_vars = self._get_vars(y_vars_name, time_freq)

        x_vars_name_raw = self.para_dict['x_vars']

        if 'x_vars_moving_average' in self.para_dict.keys():
            x_vars_name_raw1 = x_vars_name_raw + self.para_dict['x_vars_moving_average']
        else:
            x_vars_name_raw1 = x_vars_name_raw
        if 'high_order2_term_x' in self.para_dict.keys():
            x_vars_name_high_order2_ = self.para_dict['high_order2_term_x']
            x_vars_name_high_order2 = util.util.high_order_name_change(x_vars_name_high_order2_, 2)
            x_vars_name = x_vars_name_raw1 + x_vars_name_high_order2
        else:
            x_vars_name = x_vars_name_raw1

        x_vars = self._get_vars(x_vars_name, time_freq)

        data_merged = pd.DataFrame(pd.concat([x_vars, y_vars], keys=['x', 'y'], axis=1))
        data_merged_drop_na = data_merged.dropna()
        x_vars_dropna = data_merged_drop_na['x']
        y_vars_dropna = data_merged_drop_na['y']

        my_log.info('data_length_raw: {}\ndata_length_na: {}\ndata_length_dropna: {}'
                    .format(len(data_merged), len(data_merged) - len(data_merged_drop_na), len(data_merged_drop_na)))

        return y_vars_dropna, x_vars_dropna

    def _get_vars(self, vars_name, time_freq):
        assert isinstance(vars_name, list)

        my_data = self._get_data_cols(vars_name)
        # data_rolling = my_data.rolling(window=util.util.get_windows(time_scale_long=lag_scale)).mean()
        data_freq = util.util.my_resample(my_data, time_freq=time_freq)
        # data_freq = my_data.groupby(util.util.datetime2ymdstr, group_keys=False).apply(lambda x: x.resample(time_freq).apply('last').select(util.util.is_in_market_open_time))
        return data_freq

    def _get_data_cols(self, vars_name):
        data_list = []
        for var_name in vars_name:
            data_list.append(self._get_one_col(var_name))
        data_df = pd.DataFrame(pd.concat(data_list, keys=vars_name, axis=1))
        return data_df

    def _get_one_col(self, var_name):
        my_log.debug(var_name)
        data_raw = self.data_df
        time_scale_raw = data_raw.index[1] - data_raw.index[0]
        if var_name in data_raw.columns:
            data_new = data_raw[var_name]
        elif var_name == 'spread':
            data_new = data_raw['ask1'] - data_raw['bid1']
            data_new[(data_new >= self.para_dict['spread_threshold'][1]) | (data_new <= self.para_dict['spread_threshold'][0])] = np.nan
        elif var_name == 'mid_px_ret_15s':
            mid_px = (data_raw['ask1'] + data_raw['bid1'])/2
            mid_px_ret = self._get_lag_return(mid_px, raw_scale=time_scale_raw, target_scale='15s')
            data_new = mid_px_ret
        elif var_name == 'mid_px_ret':
            mid_px = (data_raw['ask1'] + data_raw['bid1'])/2
            mid_px_ret = mid_px / mid_px.shift(1) - 1
            data_new = mid_px_ret
        elif var_name == 'mid_px_ret_dummy':
            mid_px = (data_raw['ask1'] + data_raw['bid1'])/2
            mid_px_ret = mid_px / mid_px.shift(1) - 1
            data_new = pd.Series(np.where(mid_px_ret == 0, [1]*len(mid_px_ret), [0]*len(mid_px_ret)), index=mid_px.index)
        elif var_name == 'ret_sh50_15s':
            sh50_px = data_raw['price_index_sh50']
            data_new = self._get_lag_return(sh50_px, raw_scale=time_scale_raw, target_scale='15s')
        elif var_name == 'ret_sh50':
            sh50_px = data_raw['price_index_sh50']
            data_new = sh50_px / sh50_px.shift(1) - 1
        elif var_name == 'ret_index_index_future_300':
            index_future_px = data_raw['price_index_index_future_300']
            data_new = index_future_px / index_future_px.shift(1) - 1
        elif var_name == 'ret_hs300_15s':
            sh300_px = data_raw['price_index_hs300']
            data_new = self._get_lag_return(sh300_px, raw_scale=time_scale_raw, target_scale='15s')
        elif var_name == 'bid1_ret_15s':
            bid1 = data_raw['bid1']
            data_new = self._get_lag_return(bid1, raw_scale=time_scale_raw, target_scale='15s')
        elif var_name == 'ask1_ret_15s':
            ask1 = data_raw['ask1']
            data_new = self._get_lag_return(ask1, raw_scale=time_scale_raw, target_scale='15s')
        elif var_name == 'ret_hs300':
            sh300_px = data_raw['price_index_hs300']
            data_new = sh300_px / sh300_px.shift(1) - 1
        elif var_name == 'bid1_ret':
            bid1 = data_raw['bid1']
            data_new = bid1 / bid1.shift(1) - 1
        elif var_name == 'ask1_ret':
            ask1 = data_raw['ask1']
            data_new = ask1 / ask1.shift(1) - 1
        elif var_name == 'volatility_index300_60s':
            sh300_px = data_raw['price_index_hs300']
            sh300_ret = sh300_px.pct_change().fillna(method='ffill')
            data_new = sh300_ret.rolling(window=20).std()
        elif var_name == 'volatility_index50_60s':
            sh300_px = data_raw['price_index_sh50']
            sh300_ret = sh300_px.pct_change().fillna(method='ffill')
            data_new = sh300_ret.rolling(window=20).std()
        elif var_name == 'volatility_mid_px_60s':
            mid_px = (data_raw['ask1'] + data_raw['bid1'])/2
            mid_px_ret = mid_px.pct_change().fillna(method='ffill')
            data_new = mid_px_ret.rolling(window=20).std()
        elif var_name == 'bsize1_change':
            data_new = data_raw['bsize1'] - data_raw['bsize1'].shift(1)
        elif var_name == 'asize1_change':
            data_new = data_raw['asize1'] - data_raw['asize1'].shift(1)
        elif var_name == 'buy_vol_10min_intraday_pattern_20_days':
            data_vol = data_raw[['buyvolume']]
            data_vol.loc[:, 'index'] = data_vol.index
            data_vol.loc[:, 'date'] = data_vol['index'].apply(lambda x: (x.year, x.month, x.day))
            data_vol.loc[:, 'period'] = data_vol['index'].apply(util.util.in_intraday_period)
            data_vol['new_index'] = list(zip(data_vol['date'], data_vol['period']))

            vol_mean_by_date_and_period = data_vol.groupby(['date', 'period'])['buyvolume'].mean()
            vol_wide = vol_mean_by_date_and_period.unstack().sort_index()
            vol_wide_rolling_mean = vol_wide.rolling(window=20).mean().shift(1)
            vol_long = vol_wide_rolling_mean.stack()

            data_new = pd.DataFrame(vol_long[data_vol['new_index']]).set_index(data_vol.index)[0]
        elif var_name == 'sell_vol_10min_intraday_pattern_20_days':
            data_vol = data_raw[['sellvolume']]
            data_vol.loc[:, 'index'] = data_vol.index
            data_vol.loc[:, 'date'] = data_vol['index'].apply(lambda x: (x.year, x.month, x.day))
            data_vol.loc[:, 'period'] = data_vol['index'].apply(util.util.in_intraday_period)
            data_vol['new_index'] = list(zip(data_vol['date'], data_vol['period']))

            vol_mean_by_date_and_period = data_vol.groupby(['date', 'period'])['sellvolume'].mean()
            vol_wide = vol_mean_by_date_and_period.unstack().sort_index()
            vol_wide_rolling_mean = vol_wide.rolling(window=20).mean().shift(1)
            vol_long = vol_wide_rolling_mean.stack()

            data_new = pd.DataFrame(vol_long[data_vol['new_index']]).set_index(data_vol.index)[0]
        # moving average terms
        elif var_name in [
            'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]:
            var_name_prefix = '_'.join(var_name.split('_')[:-1])
            ma_days = int(re.search('(?<=mean)\d+(?=day)', var_name).group())
            data_col = data_raw[[var_name_prefix]]
            data_col.loc[:, 'index'] = data_col.index
            data_col.loc[:, 'date'] = data_col['index'].apply(lambda x: (x.year, x.month, x.day))
            data_col['new_index'] = data_col['date']

            data_mean_by_date_and_period = data_col.groupby(['date'])[var_name_prefix].mean()
            data_wide = data_mean_by_date_and_period.sort_index()
            data_wide_rolling_mean = data_wide.rolling(window=ma_days).mean().shift(1)
            data_long = data_wide_rolling_mean
            data_new = pd.DataFrame(data_long[data_col['new_index']]).set_index(data_col.index)

        elif var_name.endswith('_order2'):  # todo
            var_name_prefix = var_name[:-7]
            data_new_ = self._get_one_col(var_name_prefix)
            # data_new = data_new_.values * data_new_.values
            # data_new = pd.DataFrame(data_new, index=data_new_.index)
            data_new = data_new_ * data_new_
        else:
            my_log.error(var_name)
            raise LookupError

        return data_new

    @classmethod
    def _get_lag_return(cls, data_raw, raw_scale, target_scale):
        def lag_ret(chunk):
            c = [x for x in chunk if x != 0]
            return c[-1] / c[0] - 1 if len(c) != 0 else 0.0
        if isinstance(raw_scale, datetime.timedelta) and isinstance(target_scale, str):
            target_scale_ = pd.datetools.to_offset(target_scale).delta
            windows = int(target_scale_ / raw_scale)
            data_new = data_raw.fillna(0).rolling(window=windows).apply(lag_ret)
        else:
            my_log.error('type_error:' + str(raw_scale))
            my_log.error('type_error:' + str(target_scale))
            raise TypeError
        return data_new


class TrainingData(DataBase):
    pass


class TestingData(DataBase):
    pass


class DataRolling(DataBase):
    def __init__(self, date_begin, date_end, para_dict, training_period, testing_period, testing_demean_period=None):
        DataBase.__init__(self, date_begin, date_end, para_dict)
        self.training_period = training_period
        self.testing_period = testing_period
        self.test_demean_period = testing_demean_period

    def generating_rolling_data(self, fixed=False):

        if self.training_period == '12M' and self.testing_period == '1M':
            offset_training = pd.tseries.offsets.MonthEnd(12)  # todo
            offset_predict = pd.tseries.offsets.MonthEnd(1)
        elif self.training_period == '6M' and self.testing_period == '1M':
            offset_training = pd.tseries.offsets.MonthEnd(6)
            offset_predict = pd.tseries.offsets.MonthEnd(1)
        elif self.training_period == '1M' and self.testing_period == '1M':
            offset_training = pd.tseries.offsets.MonthEnd(1)
            offset_predict = pd.tseries.offsets.MonthEnd(1)
        else:
            my_log.error('training_period: ' + self.training_period)
            my_log.error('testing_period: ' + self.testing_period)
            raise AssertionError

        if self.test_demean_period is None:
            offset_test_demean = offset_training
        else:
            if self.test_demean_period == '12M':
                offset_test_demean = pd.tseries.offsets.MonthEnd(12)
            else:
                raise ValueError

        offset_one_day = pd.tseries.offsets.Day(1)
        keys = ['data_training', 'data_predicting', 'data_out_of_sample_demean', 'in_sample_period', 'out_of_sample_period', 'demean_period']

        my_data = self.data_df

        dates_ = pd.Series([x for x in list(my_data.index) if self.date_end >= x.strftime('%Y%m%d') >= self.date_begin])
        date_begin = dates_.iloc[0]
        date_end = dates_.iloc[-1]
        date_moving = date_begin

        training_date_begin = date_moving    # todo
        training_date_end = date_moving + offset_training

        while True:
            if not fixed:
                training_date_begin = date_moving
                training_date_end = date_moving + offset_training
                predict_date_begin = training_date_end + offset_one_day
                predict_date_end = predict_date_begin + offset_predict
                demean_date_begin = predict_date_begin - offset_one_day - offset_test_demean
                demean_date_end = predict_date_begin - offset_one_day
            else:
                # training_date_begin_ = date_moving
                training_date_end_ = date_moving + offset_training

                predict_date_begin = training_date_end_ + offset_one_day
                predict_date_end = predict_date_begin + offset_predict

                demean_date_begin = predict_date_begin - offset_one_day - offset_test_demean
                demean_date_end = predict_date_begin - offset_one_day

            if training_date_begin < date_begin or demean_date_begin < date_begin:
                pass
            else:
                if predict_date_end > date_end or training_date_end > date_end:
                    raise StopIteration

                my_log.info('rolling: {}, {}, {}, {}'.format(training_date_begin, training_date_end, predict_date_begin, predict_date_end))

                data_training_df = my_data.select(lambda x: training_date_end >= x >= training_date_begin)
                data_predicting_df = my_data.select(lambda x: predict_date_end >= x >= predict_date_begin)
                data_out_of_sample_demean_df = my_data.select(lambda x: demean_date_end >= x >= demean_date_begin)

                data_training = TrainingData(data_df=data_training_df, have_data_df=True, para_dict=self.para_dict)
                data_predicting = TestingData(data_df=data_predicting_df, have_data_df=True, para_dict=self.para_dict)
                data_out_of_sample_demean = TrainingData(data_df=data_out_of_sample_demean_df, have_data_df=True, para_dict=self.para_dict)

                in_sample_period = ''.join([training_date_begin.strftime('%Y%m%d'), '_', training_date_end.strftime('%Y%m%d')])
                out_of_sample_period = ''.join([predict_date_begin.strftime('%Y%m%d'), '_', predict_date_end.strftime('%Y%m%d')])
                demean_period = ''.join([demean_date_begin.strftime('%Y%m%d'), '_', demean_date_end.strftime('%Y%m%d')])

                to_yield = dict(list(zip(keys, [data_training, data_predicting, data_out_of_sample_demean, in_sample_period, out_of_sample_period, demean_period])))

                my_log.info('data_training: {}\ndata_predicting: {}\ndemean_period: {}'.format(in_sample_period, out_of_sample_period, demean_date_end))

                yield to_yield
            date_moving = date_moving + offset_predict
