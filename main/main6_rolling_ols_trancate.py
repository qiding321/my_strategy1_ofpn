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


def main():
    my_para = paras.paras.Paras().my_para
    description = my_para['description']
    training_period, testing_period, testing_demean_period = (my_para[col] for col in ['training_period', 'testing_period', 'testing_demean_period'])

    # ==========================output path======================
    time_now_str = util.util.get_timenow_str()
    output_path = my_path.path.market_making_result_root + time_now_str + description + '\\'

    # ==========================date=============================
    rolling_date_begin = '20130801'
    # training_date_begin = '20150101'
    rolling_date_end = '20160831'

    # =========================log================================
    my_log = log.log.log_order_flow_predict
    my_log.add_path(log_path2=output_path + 'log.log')
    my_log.info('description: %s' % description)
    my_log.info('rolling_date_begin: {}\nrolling_date_end: {}\ntraining period: {}\ntesting period: {}\ndemean period: {}'
                .format(rolling_date_begin, rolling_date_end, training_period, testing_period, testing_demean_period))
    my_log.info(util.util.dict2str(my_para))
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
        reg_data_training, normalize_funcs_useless = data_training.generate_reg_data()
        reg_data_demean_useless, normalize_funcs = data_demean.generate_reg_data()
        reg_data_testing, normalize_funcs = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs)

        assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
        assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

        # ===========================reg and predict=====================
        reg_result = reg_data_training.fit()
        reg_data_testing.add_model(reg_data_training.model, reg_data_training.paras_reg)
        predict_result = reg_data_testing.predict()

        r_sq_in_sample = util.util.cal_r_squared(y_raw=reg_data_training.y_vars.values.T[0], y_predict=reg_data_training.y_predict_insample, y_training=reg_data_training.y_vars.values.T[0])
        r_sq_out_of_sample = util.util.cal_r_squared(y_raw=reg_data_testing.y_vars.values.T[0], y_predict=reg_data_testing.predict_y.T, y_training=reg_data_training.y_vars.values.T[0])
        with open(output_file, 'a') as f_out:
            to_record = '{time_period_in_sample},{time_period_out_of_sample},{rsquared_in_sample},{rsquared_out_of_sample}\n'.format(
                time_period_in_sample=in_sample_period, time_period_out_of_sample=out_of_sample_period, rsquared_in_sample=r_sq_in_sample, rsquared_out_of_sample=r_sq_out_of_sample
            )
            f_out.write(to_record)
        time_period_name = in_sample_period + '_' + out_of_sample_period
        err_testing = reg_data_testing.get_err()
        reg_data_testing.report_err(output_path, err_testing, name=time_period_name)
        reg_data_testing.report_monthly(output_path, name_time_period=time_period_name, normalize_funcs=normalize_funcs)


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
