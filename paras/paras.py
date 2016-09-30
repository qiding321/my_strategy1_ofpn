# -*- coding: utf-8 -*-
"""
Created on 2016/8/30 10:12

@author: qiding
"""

import util.const


class Paras:
    def __init__(self):
        self.min_tick = 100

    @property
    def paras1_high_order(self):
        d = {
            'method': 'test',
            'time_freq': '15s',
            # 'time_scale_x': '15s',
            # 'time_scale_y': '15s',
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
                'ret_index_index_future_300', 'volume_index_index_future_300',
            ],
            'x_vars_moving_average': [
                'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
                'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
                'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
                'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
                'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
            ],
            # 'y_vars': ['sellvolume'],
            'y_vars': ['buyvolume'],
            'high_order2_term_x': [
                'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
                'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
                'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
                'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
                'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
            ],
            'add_const': True,
            # 'add_const': False,
            'spread_threshold': (0 * self.min_tick, 100 * self.min_tick)
        }
        return d

    @property
    def paras_after_selection_buy(self):
        d = {
            'method': 'test',
            'time_freq': '15s',
            'time_scale_x': '1min',
            'time_scale_y': '1min',
            'x_vars': [
                'asize2',
                'buyvolume',
                'sellvolume',
                'volume_index_sh50',
                'ret_index_index_future_300',
                'bsize1_change',
                ],
            # 'y_vars': ['sellvolume'],
            'y_vars': ['buyvolume'],
            'spread_threshold': (0 * self.min_tick, 100 * self.min_tick),
            'lag_terms': [  # (var_name, [lags])
                ('buyvolume', [2, 3]),
            ],
            # 'high_order2_term_x': [
            #     'buyvolume'
            # ],

        }
        return d

    @property
    def my_para(self):
        training_period = '12M'
        testing_period = '1M'
        testing_demean_period = '12M'
        reg_name = 'fixed_vars'
        # reg_name = 'model_selection'

        normalize = True
        # normalize = False
        # divided_std = True
        divided_std = False

        # method = util.const.FITTING_METHOD.ADABOOST
        method = util.const.FITTING_METHOD.OLS
        # method = util.const.FITTING_METHOD.DECTREE

        para_type = 'selected_vars'

        # para_type = 'all_vars'

        class trancate_para:
            trancate = 'True'
            trancate_method = 'mean_std'
            # trancate_method = 'winsorize'
            trancate_window = 20
            trancate_std = 3
            # trancate_winsorize = 0.99

            trancate_vars = ['buyvolume', 'sellvolume']

            @classmethod
            def descrip(cls):
                if cls.trancate:
                    s = '_trancate_method_{}_period_{}_std_{}'.format(
                        cls.trancate_method, cls.trancate_window, cls.trancate_std
                    )
                else:
                    s = '_not_trancate'
                return s

        class decision_tree_para:
            decision_tree = False
            # decision_tree_depth = 5
            # decision_tree_depth = 10
            decision_tree_depth = None

            @classmethod
            def descrip(cls):
                if cls.decision_tree:
                    s = '_decision_depth_{}'.format(cls.decision_tree_depth)
                else:
                    s = ''
                return s

        # ==========================description=======================
        # description = 'rolling_error_decomposition_{}_predict_{}_normalized_by_{}_add_ma_add_high_order'.format(training_period, testing_period, testing_demean_period)
        # description = 'rolling_{}_{}_predict_{}_normalized_by_{}_all_vars_{}_1min_depth{}'.format(method, training_period, testing_period, testing_demean_period,
        #                                                                                           'divstd' if divided_std else 'notdivstd', decision_tree_depth)
        description = 'rolling_{}_{}_{}_predict_{}_normalized_by_{}_{}_{}_1min{trancate_descrip}{decision_tree_descrip}'.format(
            reg_name,
            method, training_period, testing_period, testing_demean_period, para_type,
            'divstd' if divided_std else 'notdivstd',
            trancate_descrip=trancate_para.descrip(),
            decision_tree_descrip=decision_tree_para.descrip()
        )
        # description = 'rolling_decision_tree_{}_predict_{}_normalized_by_{}_selected_vars_divstd_1min_depth5'.format(training_period, testing_period, testing_demean_period)

        # ==========================paras============================
        if para_type == 'selected_vars':
            this_para = self.paras_after_selection_buy
        elif para_type == 'all_vars':
            this_para = self.paras1_high_order
        else:
            print('unknow para type')
            raise ValueError

        this_para['training_period'] = training_period
        this_para['testing_period'] = testing_period
        this_para['testing_demean_period'] = testing_demean_period

        this_para['add_const'] = True
        this_para['decision_tree'] = decision_tree_para
        this_para['para_type'] = para_type
        this_para['method'] = method
        this_para['normalize'] = normalize
        this_para['divided_std'] = divided_std
        this_para['trancate_para'] = trancate_para
        this_para['description'] = description

        this_para['reg_name'] = reg_name

        return this_para
