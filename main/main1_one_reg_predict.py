# -*- coding: utf-8 -*-
"""
Created on 2016/9/16 15:24

@author: qiding
"""

import data.data
import data.reg_data
import log.log
import paras.paras
import util.const
import util.util


def main():

    # ==========================date=============================
    training_date_begin = '20140801'
    training_date_begin = '20150101'
    training_date_end = '20150131'
    testing_date_begin = '20150201'
    testing_date_end = '20150231'

    # ==========================description=======================
    description = 'one_reg_result_and_analysis'  # todo

    # ==========================paras============================
    my_para = paras.paras.Paras().paras1

    # =========================log================================
    my_log = log.log.log_order_flow_predict
    my_log.info('description: %s' % description)
    my_log.info('training_date_begin:{}\ntraining_date_end:{}\ntesting_date_begin:{}\ntesting_dat_end:{}\n'
                .format(training_date_begin, training_date_end, testing_date_begin, testing_date_end))
    my_log.info(util.util.dict2str(my_para))

    # ============================loading data====================
    my_log.info('data begin')
    data_training = data.data.TrainingData(training_date_begin, training_date_end, my_para)
    data_testing = data.data.TestingData(testing_date_begin, testing_date_end, my_para)
    my_log.info('data end')

    # ============================normalize data=================
    reg_data_training, normalize_funcs = data_training.generate_reg_data()
    reg_data_testing, normalize_funcs = data_testing.generate_reg_data(normalize_funcs)

    assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
    assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

    # ===========================reg and predict=====================
    reg_result = reg_data_training.fit()
    print(reg_result)
    reg_data_testing.add_model(reg_data_training.model, reg_data_training.paras_reg)
    predict_result = reg_data_testing.predict()
    print(predict_result)


if __name__ == '__main__':
    main()
