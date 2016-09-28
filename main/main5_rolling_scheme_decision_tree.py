# -*- coding: utf-8 -*-
"""
Created on 2016/9/28 10:15

@author: qiding
"""

import data.data
import data.reg_data
import log.log
import my_path.path
import paras.paras
import util.const
import util.util

# ==========================description=======================
training_period = '12M'
testing_period = '1M'
testing_demean_period = '12M'
normalize = True
# normalize = False
divided_std = True

method = util.const.FITTING_METHOD.ADABOOST
# method = util.const.FITTING_METHOD.OLS
# method = util.const.FITTING_METHOD.DECTREE

para_type = 'selected_vars'
# para_type = 'all_vars'

decision_tree_depth = 5
# decision_tree_depth = 10

# description = 'rolling_error_decomposition_{}_predict_{}_normalized_by_{}_add_ma_add_high_order'.format(training_period, testing_period, testing_demean_period)
description = 'rolling_{}_{}_predict_{}_normalized_by_{}_all_vars_{}_1min_depth{}'.format(method, training_period, testing_period, testing_demean_period,
                                                                                          'divstd' if divided_std else 'notdivstd', decision_tree_depth)
# description = 'rolling_decision_tree_{}_predict_{}_normalized_by_{}_selected_vars_divstd_1min_depth5'.format(training_period, testing_period, testing_demean_period)


# ==========================output path======================
time_now_str = util.util.get_timenow_str()
output_path = my_path.path.market_making_result_root + time_now_str + description + '\\'


def main():
    # ==========================date=============================
    rolling_date_begin = '20130801'
    # training_date_begin = '20150101'
    rolling_date_end = '20160831'

    # ==========================paras============================
    # my_para = paras.paras.Paras().paras_after_selection  # todo
    # my_para = paras.paras.Paras().paras1_high_order  # todo
    # my_para = paras.paras.Paras().paras_after_selection  # todo
    # my_para = paras.paras.Paras().paras_neat_buy

    if para_type == 'selected_vars':
        my_para = paras.paras.Paras().paras_after_selection
    elif para_type == 'all_vars':
        my_para = paras.paras.Paras().paras1_high_order
    else:
        print('unknow para type')
        raise ValueError

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

    # output_file
    output_file = output_path + 'r_squared_record.csv'
    with open(output_file, 'w') as f_out:
        f_out.write('time_period_in_sample,time_period_out_of_sample,rsquared_in_sample,rsquared_out_of_sample\n')

    # ============================rolling==========================
    for data_rolling_once in data_rolling.generating_rolling_data():
        # ============================normalize data=================
        data_training, data_predicting, data_demean, in_sample_period, out_of_sample_period = [
            data_rolling_once[col] for col in ['data_training', 'data_predicting', 'data_out_of_sample_demean', 'in_sample_period', 'out_of_sample_period']
            ]
        assert isinstance(data_training, data.data.TrainingData) and isinstance(data_predicting, data.data.TestingData)
        reg_data_training, normalize_funcs_useless = data_training.generate_reg_data(normalize=normalize)
        reg_data_demean_useless, normalize_funcs = data_demean.generate_reg_data(normalize=normalize)
        reg_data_testing, normalize_funcs = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs, normalize=normalize, divided_std=divided_std)

        assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
        assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

        # ===========================reg and predict=====================
        reg_result = reg_data_training.fit(add_const=add_const, method=method, decision_tree_depth=decision_tree_depth)
        reg_data_testing.add_model(reg_data_training.model, reg_data_training.paras, method=method)
        predict_result = reg_data_testing.predict(add_const=add_const, method=method)

        r_sq_in_sample = util.util.cal_r_squared(y_raw=reg_data_training.y_vars.values.T[0], y_predict=reg_data_training.y_predict_insample, y_training=reg_data_training.y_vars.values.T[0])
        r_sq_out_of_sample = util.util.cal_r_squared(y_raw=reg_data_testing.y_vars.values.T[0], y_predict=reg_data_testing.predict_y.T, y_training=reg_data_training.y_vars.values.T[0])
        with open(output_file, 'a') as f_out:
            to_record = '{time_period_in_sample},{time_period_out_of_sample},{rsquared_in_sample},{rsquared_out_of_sample}\n'.format(
                time_period_in_sample=in_sample_period, time_period_out_of_sample=out_of_sample_period, rsquared_in_sample=r_sq_in_sample, rsquared_out_of_sample=r_sq_out_of_sample
            )
            f_out.write(to_record)


if __name__ == '__main__':
    # import cProfile
    #
    # cprofile_path = output_path + 'cProfile'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # cProfile.run('main()', cprofile_path)
    #
    # import pstats
    #
    # p = pstats.Stats(cprofile_path)
    # p.sort_stats('cumulative').print_stats()

    main()
