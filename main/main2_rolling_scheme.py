# -*- coding: utf-8 -*-
"""
Created on 2016/9/16 15:24

@author: qiding
"""

import log.log
import paras.paras
import data.data
import data.reg_data
import util.const
import util.util
import my_path.path


def main():

    # ==========================date=============================
    rolling_date_begin = '20130801'
    # training_date_begin = '20150101'
    rolling_date_end = '20160831'

    # ==========================description=======================
    # description = 'rolling_new_one_year_not_normalized_neat_buy'  # todo
    # description = 'rolling_one_year_normalized_neat_buy'  # todo
    # description = 'rolling_new_one_year_not_normalized_buy'  # todo
    # description = 'rolling_one_year_normalized_buy'  # todo
    description = 'rolling_one_year_normalized_buy_selected_vars'  # todo
    # description = 'test'  # todo
    training_period = '12M'
    testing_period = '1M'
    normalize = True
    # normalize = False

    # ==========================paras============================
    # my_para = paras.paras.Paras().paras1  # todo
    my_para = paras.paras.Paras().paras_after_selection  # todo
    # my_para = paras.paras.Paras().paras_neat_buy
    add_const = True if 'add_const' not in my_para.keys() else my_para['add_const']

    # ==========================output path======================
    time_now_str = util.util.get_timenow_str()
    output_path = my_path.path.market_making_result_root + time_now_str + description + '\\'

    # =========================log================================
    my_log = log.log.log_order_flow_predict
    my_log.info('description: %s' % description)
    my_log.info('rolling_date_begin: {}\nrolling_date_end: {}\ntraining period: {}\ntesting period: {}'
                .format(rolling_date_begin, rolling_date_end, training_period, testing_period))
    my_log.info(util.util.dict2str(my_para))
    my_log.info('output path: {}'.format(output_path))

    # ============================loading data====================
    my_log.info('data begin')
    data_rolling = data.data.DataRolling(rolling_date_begin, rolling_date_end, my_para, training_period, testing_period)
    my_log.info('data end')

    # ============================rolling==========================
    for data_rolling_once in data_rolling.generating_rolling_data():
        # ============================normalize data=================
        data_training, data_predicting, in_sample_period, out_of_sample_period = [data_rolling_once[col] for col in ['data_training', 'data_predicting', 'in_sample_period', 'out_of_sample_period']]
        reg_data_training, normalize_funcs = data_training.generate_reg_data(normalize=normalize)
        reg_data_testing, normalize_funcs = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs, normalize=normalize)

        assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
        assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

        # ===========================reg and predict=====================
        reg_result = reg_data_training.fit(add_const=add_const)
        # print(reg_result)
        reg_data_testing.add_model(reg_data_training.model, reg_data_training.paras)
        predict_result = reg_data_testing.predict(add_const=add_const)
        # print(predict_result)

        name = in_sample_period + '_' + out_of_sample_period
        err_testing = reg_data_testing.get_err(add_const=add_const)
        reg_data_testing.report_err(output_path, err_testing, name=name)



if __name__ == '__main__':
    main()
