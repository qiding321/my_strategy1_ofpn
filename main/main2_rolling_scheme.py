# -*- coding: utf-8 -*-
"""
Created on 2016/9/16 15:24

@author: qiding
"""

import os

import data.data
import data.reg_data
import log.log
import my_path.path
import paras.paras
import util.const
import util.util

# ==========================description=======================
# description = 'rolling_new_one_year_not_normalized_neat_buy'  # todo
# description = 'rolling_one_year_normalized_neat_buy'  # todo
# description = 'rolling_new_one_year_not_normalized_buy'  # todo
# description = 'rolling_one_year_normalized_buy_68vars_add_const'  # todo
# description = 'rolling_one_year_normalized_buy_68vars_not_add_const'  # todo
# description = 'rolling_one_year_normalized_buy_selected_vars'  # todo
# description = 'test'  # todo

training_period = '12M'
testing_period = '1M'
testing_demean_period = '12M'
normalize = True
# normalize = False
# description = 'rolling_error_decomposition_{}_predict_{}_normalized_by_{}_add_ma_add_high_order'.format(training_period, testing_period, testing_demean_period)
description = 'rolling_error_decomposition_{}_predict_{}_normalized_by_{}_add_ma'.format(training_period, testing_period, testing_demean_period)

# ==========================output path======================
time_now_str = util.util.get_timenow_str()
output_path = my_path.path.market_making_result_root + time_now_str + description + '\\'


def main():

    # ==========================date=============================
    rolling_date_begin = '20130801'
    # training_date_begin = '20150101'
    rolling_date_end = '20160831'

    # ==========================paras============================
    my_para = paras.paras.Paras().paras1  # todo
    # my_para = paras.paras.Paras().paras1_high_order  # todo
    # my_para = paras.paras.Paras().paras_after_selection  # todo
    # my_para = paras.paras.Paras().paras_neat_buy
    add_const = True if 'add_const' not in my_para.keys() else my_para['add_const']

    # =========================log================================
    my_log = log.log.log_order_flow_predict
    my_log.add_path(log_path2=output_path + 'log.log')
    my_log.info('description: %s' % description)
    my_log.info('rolling_date_begin: {}\nrolling_date_end: {}\ntraining period: {}\ntesting period: {}\ndemean period: {}'
                .format(rolling_date_begin, rolling_date_end, training_period, testing_period, testing_demean_period))
    my_log.info(util.util.dict2str(my_para))
    my_log.info('normalize: {}\nadd_const: {}'.format(normalize, add_const))
    my_log.info('output path: {}'.format(output_path))

    # ============================loading data====================
    my_log.info('data begin')
    data_rolling = data.data.DataRolling(rolling_date_begin, rolling_date_end, my_para, training_period, testing_period, testing_demean_period=testing_demean_period)
    my_log.info('data end')

    # ============================rolling==========================
    for data_rolling_once in data_rolling.generating_rolling_data():
        # ============================normalize data=================
        data_training, data_predicting, data_demean, in_sample_period, out_of_sample_period = [
            data_rolling_once[col] for col in ['data_training', 'data_predicting', 'data_out_of_sample_demean', 'in_sample_period', 'out_of_sample_period']
            ]
        assert isinstance(data_training, data.data.TrainingData) and isinstance(data_predicting, data.data.TestingData)
        reg_data_training, normalize_funcs_useless = data_training.generate_reg_data(normalize=normalize)
        reg_data_demean_useless, normalize_funcs = data_demean.generate_reg_data(normalize=normalize)
        reg_data_testing, normalize_funcs = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs, normalize=normalize)

        assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
        assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

        # ===========================reg and predict=====================
        reg_result = reg_data_training.fit(add_const=add_const)
        # print(reg_result)
        reg_data_testing.add_model(reg_data_training.model, reg_data_training.paras_reg)
        predict_result = reg_data_testing.predict(add_const=add_const)
        # print(predict_result)

        name = in_sample_period + '_' + out_of_sample_period
        err_testing = reg_data_testing.get_err(add_const=add_const)
        reg_data_testing.report_err(output_path, err_testing, name=name)


if __name__ == '__main__':
    import cProfile

    cprofile_path = output_path + 'cProfile'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cProfile.run('main()', cprofile_path)

    import pstats

    p = pstats.Stats(cprofile_path)
    p.sort_stats('cumulative').print_stats()


    # main()
