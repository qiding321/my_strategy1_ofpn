# -*- coding: utf-8 -*-
"""
Created on 2016/9/28 10:15

@author: qiding
"""

import os

import pandas as pd

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

    model_selection_result = dict()

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
        if my_para['reg_name'] == 'fixed_vars':
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

        # ===========================model selection=====================
        if my_para['reg_name'] == 'model_selection':
            model_selection_result_this_time_period = dict()
            name = in_sample_period + '_' + out_of_sample_period

            vars_left = reg_data_training.x_var_names
            num_init = len(vars_left)

            for num_now in range(num_init, 0, -1):
                my_log.info('model_selection: ' + str(num_now))
                if check_valid2(num_now - 1):
                    pass
                else:
                    break

                for reg_count, reg_data_vars_iter in enumerate(reg_data_training.vars_iteration_model_selection(vars_left)):
                    reg_data_vars_len = reg_data_vars_iter.num_of_x_vars
                    assert reg_data_vars_len == num_now - 1

                    assert isinstance(reg_data_vars_iter, data.reg_data.RegDataTraining)
                    reg_data_testing_vars_iter = reg_data_testing.generate_vars_model_selection(reg_data_vars_iter.x_var_names)
                    assert isinstance(reg_data_testing_vars_iter, data.reg_data.RegDataTest)

                    # regress proceed
                    reg_result = reg_data_vars_iter.fit()
                    reg_data_testing_vars_iter.add_model(reg_data_vars_iter.model, reg_data_vars_iter.paras_reg)
                    # predict_result = reg_data_testing_vars_iter.predict(add_const=add_const)

                    err_dict = reg_data_testing_vars_iter.get_err()
                    predict_result2 = (err_dict['rsquared_out_of_sample'], err_dict['sse'], err_dict['ssr'])  # (rsquared_, mse_, msr_)

                    # record and judge
                    if reg_data_vars_len not in model_selection_result_this_time_period.keys():
                        model_selection_result_this_time_period[reg_data_vars_len] = dict()
                    model_selection_result_this_time_period[reg_data_vars_len][tuple(sorted(reg_data_vars_iter.x_var_names))] = predict_result2
                    path_ = _get_detail_record_path(output_path=output_path, var_num=reg_data_vars_len, reg_count=reg_count, test_sample_period=out_of_sample_period)
                    util.util.record_result(
                        to_record_str=reg_data_vars_iter.result_record_str() + '\n' + reg_data_testing_vars_iter.result_record_str(),
                        to_record_path=path_
                    )

                vars_left = sorted([(v, k) for k, v in model_selection_result_this_time_period[num_now - 1].items()], key=lambda x: x[0])[-1][1]

            model_selection_result[name] = model_selection_result_this_time_period

    if my_para['reg_name'] == 'model_selection':
        my_log.info('output: {}'.format(output_path))
        result_df = model_selection_result_unstack(model_selection_result)
        result_best = result_df.groupby('time_period', group_keys=False).apply(lambda x: x.sort_values(by=['rsquared_oos']).iloc[-1, :])
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)
        result_df.to_csv(output_path + 'result_all.csv', index=None)
        result_best.to_csv(output_path + 'result_best.csv', index=None)


def _get_detail_record_path(output_path, var_num, reg_count, test_sample_period):
    this_path_ = '{output_path}detail\\{test_sample_period}\\var_num_{var_num}\\{reg_count}\\'.format(output_path=output_path, var_num=var_num, reg_count=reg_count,
                                                                                                      test_sample_period=test_sample_period)
    return this_path_


def model_selection_result_unstack(model_selection_result):
    col_names = ['time_period', 'model_length', 'var_names', 'rsquared_oos', 'mse', 'msr']
    time_period = []
    model_length = []
    var_names = []
    rsquared_oos = []
    mse = []
    msr = []
    for time_period_, model_selection_result_this_time_period in model_selection_result.items():
        for reg_data_vars_len, v_ in model_selection_result_this_time_period.items():
            for x_names, (rsquared_, mse_, msr_) in v_.items():
                time_period.append(time_period_)
                model_length.append(reg_data_vars_len)
                var_names.append(x_names)
                rsquared_oos.append(rsquared_)
                mse.append(mse_)
                msr.append(msr_)
    df_to_ret = pd.DataFrame([time_period, model_length, var_names, rsquared_oos, mse, msr], index=col_names).T
    return df_to_ret


def check_valid2(reg_data_vars_len):  # todo check
    if reg_data_vars_len < 1:
        return False
    else:
        return True


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
