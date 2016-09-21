# -*- coding: utf-8 -*-
"""
Created on 2016/8/30 10:12

@author: qiding
"""


class Paras:
    def __init__(self):
        self.min_tick = 100

    @property
    def paras1(self):
        d = {
            'method': 'test',
            'time_freq': '15s',
            'time_scale_x': '1min',
            'time_scale_y': '1min',
            'x_vars': [
                'bsize1', 'asize1',
                'bsize2', 'asize2',
                'bsize3', 'asize3',
                'spread',
                'sellvolume', 'buyvolume',
                'volume_index_sh50', 'volume_index_hs300', 'ret_sh50', 'ret_hs300',
                'mid_px_ret', 'mid_px_ret_dummy',  # 'bid1_ret', 'ask1_ret',
                'bsize1_change', 'asize1_change',
                # 'volatility_index300_60s', 'volatility_mid_px_60s',
                'volatility_index300_60s', 'volatility_index50_60s', 'volatility_mid_px_60s',
                'buy_vol_5min_intraday_pattern_20_days', 'sell_vol_5min_intraday_pattern_20_days',
                'ret_index_index_future_300', 'volume_index_index_future_300',
                ],
            # 'y_vars': ['sellvolume'],
            'y_vars': ['buyvolume'],
            'spread_threshold': (0 * self.min_tick, 100 * self.min_tick)
        }
        return d

    @property
    def paras1_not_const(self):
        d = {
            'method': 'test',
            'time_freq': '15s',
            'time_scale_x': '1min',
            'time_scale_y': '1min',
            'x_vars': [
                'bsize1', 'asize1',
                'bsize2', 'asize2',
                'bsize3', 'asize3',
                'spread',
                'sellvolume', 'buyvolume',
                'volume_index_sh50', 'volume_index_hs300', 'ret_sh50', 'ret_hs300',
                'mid_px_ret', 'mid_px_ret_dummy',  # 'bid1_ret', 'ask1_ret',
                'bsize1_change', 'asize1_change',
                # 'volatility_index300_60s', 'volatility_mid_px_60s',
                'volatility_index300_60s', 'volatility_index50_60s', 'volatility_mid_px_60s',
                'buy_vol_5min_intraday_pattern_20_days', 'sell_vol_5min_intraday_pattern_20_days',
                'ret_index_index_future_300', 'volume_index_index_future_300',
                ],
            # 'y_vars': ['sellvolume'],
            'y_vars': ['buyvolume'],
            'add_const': False,
            'spread_threshold': (0 * self.min_tick, 100 * self.min_tick)
        }
        return d

    @property
    def paras1_sell(self):
        d = {
            'method': 'test',
            'time_freq': '15s',
            'time_scale_x': '1min',
            'time_scale_y': '1min',
            'x_vars': [
                'bsize1', 'asize1',
                'bsize2', 'asize2',
                'bsize3', 'asize3',
                'spread',
                'sellvolume', 'buyvolume',
                'volume_index_sh50', 'volume_index_hs300', 'ret_sh50', 'ret_hs300',
                'mid_px_ret', 'mid_px_ret_dummy',  # 'bid1_ret', 'ask1_ret',
                'bsize1_change', 'asize1_change',
                # 'volatility_index300_60s', 'volatility_mid_px_60s',
                'volatility_index300_60s', 'volatility_index50_60s', 'volatility_mid_px_60s',
                'buy_vol_5min_intraday_pattern_20_days', 'sell_vol_5min_intraday_pattern_20_days',
                'ret_index_index_future_300', 'volume_index_index_future_300',
                ],
            'y_vars': ['sellvolume'],
            # 'y_vars': ['buyvolume'],
            'spread_threshold': (0 * self.min_tick, 100 * self.min_tick)
        }
        return d

    @property
    def paras_after_selection(self):
        d = {
            'method': 'test',
            'time_freq': '15s',
            'time_scale_x': '1min',
            'time_scale_y': '1min',
            'x_vars': [
                'bsize1', 'asize1',
                'bsize2', 'asize2',
                'bsize3', 'asize3',
                'bsize1_change', 'asize1_change',
                'sellvolume', 'buyvolume',
                'mid_px_ret', 'mid_px_ret_dummy',
                'volume_index_sh50',
                'volatility_index300_60s',
                'ret_index_index_future_300',
                'buy_vol_5min_intraday_pattern_20_days',
                ],
            # 'y_vars': ['sellvolume'],
            'y_vars': ['buyvolume'],
            'spread_threshold': (0 * self.min_tick, 100 * self.min_tick),
            'lag_terms': [  # (var_name, [lags])
                ('asize1_change', [1, 2, 3, 4, 5]),
                ('bsize1_change', [1, 2, 3, 4, 5]),
                ('buyvolume', [1, 2, 3]),
                ('sellvolume', [1, 2, 3]),
            ]
        }
        return d

    @property
    def paras_neat_buy(self):
        d = {
            'method': 'test',
            'time_freq': '15s',
            'time_scale_x': '1min',
            'time_scale_y': '1min',
            'x_vars': [
                'asize1',
                'asize2',
                'asize3',
                'spread',
                'buyvolume',
                'volume_index_sh50', 'ret_hs300',
                'mid_px_ret_dummy',
                'volatility_index300_60s', 'volatility_mid_px_60s',
                'buy_vol_5min_intraday_pattern_20_days',
                ],
            # 'y_vars': ['sellvolume'],
            'y_vars': ['buyvolume'],

            'spread_threshold': (0 * self.min_tick, 100 * self.min_tick),
        }
        return d


