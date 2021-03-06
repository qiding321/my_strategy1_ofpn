# -*- coding: utf-8 -*-
"""
Created on 2016/9/16 16:43

@author: qiding
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import log.log
import util.const
import util.logit_wrapper
import util.util

my_log = log.log.log_order_flow_predict


class RegDataTraining:
    def __init__(self, x_vars, y_vars, paras_model):
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.num_of_x_vars = len(x_vars.columns)
        self.x_var_names = x_vars.columns
        self.paras_reg = None
        self.paras_model = paras_model
        self.model = None
        self.y_predict_insample = None

    def fit(self):
        add_const = self.paras_model['add_const']
        method = self.paras_model['method']

        if method == util.const.FITTING_METHOD.OLS:
            if add_const:
                x_new = sm.add_constant(self.x_vars)
            else:
                x_new = self.x_vars
            self.model = sm.OLS(self.y_vars, x_new, hasconst=add_const)
            self.paras_reg = self.model.fit()
            self.y_predict_insample = self.model.predict(exog=x_new, params=self.paras_reg.params)
            return self.paras_reg.rsquared
        elif method == util.const.FITTING_METHOD.DECTREE:
            decision_tree_depth = self.paras_model['decision_tree'].decision_tree_depth
            self.model = DecisionTreeClassifier(max_depth=decision_tree_depth)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict_insample = y_predict_insample
        elif method == util.const.FITTING_METHOD.DECTREEREG:
            decision_tree_depth = self.paras_model['decision_tree'].decision_tree_depth
            self.model = DecisionTreeRegressor(max_depth=decision_tree_depth)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict_insample = y_predict_insample
        elif method == util.const.FITTING_METHOD.ADABOOST:
            decision_tree_depth = self.paras_model['decision_tree'].decision_tree_depth
            rng = np.random.RandomState(1)
            self.model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=decision_tree_depth), n_estimators=300, random_state=rng)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict_insample = y_predict_insample
        elif method == util.const.FITTING_METHOD.LOGIT:
            self.model = util.logit_wrapper.LogitWrapper(endog=self.y_vars, exog=self.x_vars)
            self.paras_reg, separator = self.model.fit()
            y_predict_insample = self.model.predict(exog=self.x_vars, params=self.paras_reg.params, separator=separator)
            self.y_predict_insample = y_predict_insample
        else:
            my_log.error('reg_method not found: {}'.format(method))
            raise ValueError

    def vars_iteration_model_selection(self, vars_left):
        for drop_var in vars_left:
            vars_new = [var_ for var_ in vars_left if var_ != drop_var]
            x_vars = self.x_vars[vars_new]
            y_vars = self.y_vars
            reg_data_new = RegDataTraining(x_vars=x_vars, y_vars=y_vars, paras_model=self.paras_model)
            yield reg_data_new

    def result_record_str(self):
        coef = self.paras_reg.params
        tvalues = self.paras_reg.tvalues
        df = pd.concat([coef, tvalues], keys=['coef', 'tvalues'], axis=1).T
        to_ret = util.util.pandas2str(df, title='training_result')

        to_ret += 'rsquared,{}'.format(self.paras_reg.rsquared)

        return to_ret


class RegDataTest:
    def __init__(self, x_vars, y_vars, paras_model):
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.model = None
        self.paras_reg = None
        self.paras_model = paras_model
        self.predict_y = None
        self.err_dict = None

    def add_model(self, model=None, paras=None):
        self.model = model
        self.paras_reg = paras

    def predict(self):  # todo

        add_const = self.paras_model['add_const']
        method = self.paras_model['method']

        if method == util.const.FITTING_METHOD.OLS:
            if self.predict_y is None:
                if add_const:
                    data_predict = self.model.predict(exog=sm.add_constant(self.x_vars), params=self.paras_reg.params)
                else:
                    data_predict = self.model.predict(exog=self.x_vars, params=self.paras_reg.params)
                self.predict_y = data_predict
            else:
                data_predict = self.predict_y
            ssr = (pd.DataFrame(data_predict) - pd.DataFrame(self.y_vars.values)).values
            sse = (pd.DataFrame(self.y_vars.values) - self.model.endog.mean()).values  # for y_mean_in_sample, new
            # sse = (pd.DataFrame(self.y_vars.values) - pd.DataFrame(self.y_vars.values).mean()).values  # for y_mean_out_of_sample, old
            rsquared_out_of_sample = 1 - (ssr * ssr).sum() / (sse * sse).sum()
            return rsquared_out_of_sample
        elif method == util.const.FITTING_METHOD.DECTREE or method == util.const.FITTING_METHOD.ADABOOST or method == util.const.FITTING_METHOD.DECTREEREG:
            if self.predict_y is None:
                data_predict = self.model.predict(self.x_vars)
                self.predict_y = data_predict
            else:
                data_predict = self.predict_y
        elif method == util.const.FITTING_METHOD.LOGIT:
            if self.predict_y is None:
                if add_const:
                    data_predict = self.model.predict(exog=sm.add_constant(self.x_vars), params=self.paras_reg.params)
                else:
                    data_predict = self.model.predict(exog=self.x_vars, params=self.paras_reg.params)
                self.predict_y = data_predict
            else:
                data_predict = self.predict_y
            ssr = (pd.DataFrame(data_predict) - pd.DataFrame(self.y_vars.values)).values
            sse = (pd.DataFrame(self.y_vars.values) - self.model.endog.mean()).values  # for y_mean_in_sample, new
            # sse = (pd.DataFrame(self.y_vars.values) - pd.DataFrame(self.y_vars.values).mean()).values  # for y_mean_out_of_sample, old
            rsquared_out_of_sample = 1 - (ssr * ssr).sum() / (sse * sse).sum()
            return rsquared_out_of_sample
        else:
            my_log.error('method not found: {}'.format(method))
            raise ValueError

    def get_err(self):

        add_const = self.paras_model['add_const']

        y_ = self.y_vars.values

        if self.predict_y is None:
            if add_const:
                data_predict = self.model.predict(exog=sm.add_constant(self.x_vars), params=self.paras_reg.params)
            else:
                data_predict = self.model.predict(exog=self.x_vars, params=self.paras_reg.params)
            self.predict_y = data_predict
        else:
            data_predict = self.predict_y
        ssr = (pd.DataFrame(data_predict) - pd.DataFrame(self.y_vars.values)).values
        sse = (pd.DataFrame(self.y_vars.values) - self.model.endog.mean()).values  # for y_mean_in_sample, new
        sse_by_oos_mean = (pd.DataFrame(self.y_vars.values) - pd.DataFrame(self.y_vars.values).mean()).values  # for y_mean_out_of_sample, old
        rsquared_out_of_sample = 1 - (ssr * ssr).sum()/(sse * sse).sum()
        rsquared_out_of_sample_by_oos_mean = 1 - (ssr * ssr).sum()/(sse_by_oos_mean * sse_by_oos_mean).sum()
        var_y = y_.var()
        var_y_predict = data_predict.var()
        bias_squared = ((data_predict - y_.mean())*(data_predict - y_.mean())).mean()
        bias_mean = data_predict.mean() - y_.mean()
        cov_y_y_predict_multiplied_by_minus_2 = -2 * np.cov([y_.T[0], data_predict])[0, 1]

        ret = {
            'ssr': (ssr*ssr).mean(),
            'sse': (sse*sse).mean(),
            'sse_by_oos_mean': (sse_by_oos_mean * sse_by_oos_mean).mean(),
            'variance_x': self.x_vars.var(),
            'variance_x_contribution': self.x_vars.var() * (self.paras_reg.params * self.paras_reg.params),
            'rsquared_out_of_sample': rsquared_out_of_sample,
            'rsquared_out_of_sample_by_oos_mean': rsquared_out_of_sample_by_oos_mean,
            'var_y': var_y,
            'var_y_predict': var_y_predict,
            'bias_squared': bias_squared,
            'bias_mean': bias_mean,
            'cov_y_y_predict_multiplied_by_minus_2': cov_y_y_predict_multiplied_by_minus_2,
        }

        self.err_dict = ret

        return ret

    def result_record_str(self):
        if self.err_dict is None:
            err_dict = self.get_err()  # todo, note that "add const" is default to be True
            self.err_dict = err_dict
        else:
            err_dict = self.err_dict
        mse = err_dict['sse']
        msr = err_dict['ssr']
        rsquared = err_dict['rsquared_out_of_sample']
        rsquared_by_oos_mean = err_dict['rsquared_out_of_sample_by_oos_mean']

        to_ret_str = '\n'
        title = 'predicting_result'
        to_ret_str += title
        to_ret_str += '\n'
        to_ret_str += 'mse,{}\nmsr,{}\nrsquared,{}\nrsquared_by_oos_mean,{}'.format(mse, msr, rsquared, rsquared_by_oos_mean)

        return to_ret_str

    def generate_vars_model_selection(self, x_var_names):
        x_vars = self.x_vars[x_var_names]
        y_vars = self.y_vars
        reg_data_new = RegDataTest(x_vars=x_vars, y_vars=y_vars, paras_model=self.paras_model)
        return reg_data_new

    def report_err(self, output_path, err_dict, name):  # todo
        assert isinstance(err_dict, dict)
        file_name = 'err_record.csv'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        new_dict = {}
        for k, v in err_dict.items():
            if k == 'variance_x':
                for k_, v_ in v.items():
                    new_dict[k_+'var_x'] = v_
            elif k == 'variance_x_contribution':
                for k_, v_ in v.items():
                    new_dict[k_+'var_contrb_x'] = v_
            elif k == 'rsquared_out_of_sample' or k == 'rsquared_out_of_sample_by_oos_mean':
                new_dict[k] = v
            elif k == 'ssr' or k == 'sse' or k == 'sse_by_oos_mean':
                new_dict[k] = v
            else:
                new_dict[k] = v
        new_df = pd.DataFrame(pd.Series(new_dict), columns=[name])

        if file_name in os.listdir(output_path):
            data_exist = pd.read_csv(output_path + file_name, index_col=[0])
            data_to_rcd = pd.merge(new_df, data_exist, left_index=True, right_index=True)
        else:
            data_to_rcd = new_df

        data_to_rcd.sort_index(axis=1).to_csv(output_path + file_name)

    def report_accuracy(self, output_path, name):
        file_name = 'accuracy_record.csv'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        data_raw = pd.Series(self.y_vars)
        data_predict = pd.Series(self.predict_y)

        def _get_len(a_, b_):
            s_ = (data_raw == a_) & (data_predict == b_)
            return len(s_[s_])

        len_11 = _get_len(1, 1)
        len_10 = _get_len(1, 0)
        len_01 = _get_len(0, 1)
        len_00 = _get_len(0, 0)

        if file_name in os.listdir(output_path):
            pass
        else:
            to_rcd = ','.join(['', 'predicted_jump', 'predicted_nut_jump', 'not_predict_jump', 'not_predict_not_jump'])
            my_log.info(to_rcd)
            with open(output_path + file_name, 'a') as f_out:
                f_out.write(to_rcd)

        to_rcd = '{},{},{},{},{}\n'.format(name, len_11, len_00, len_10, len_01)
        my_log.info(to_rcd)
        with open(output_path + file_name, 'a') as f_out:
            f_out.write(to_rcd)

    def report_monthly(self, output_path, name_time_period, normalize_funcs, normalize_funcs_training):
        this_path = output_path + name_time_period + '\\'
        if os.path.exists(this_path):
            pass
        else:
            os.makedirs(this_path)

        # figure
        self._plt_predict_volume(output_path=this_path, normalize_funcs=normalize_funcs)

        # daily rsquared and mse and msr
        self._daily_rsquared_output(output_path=this_path, normalize_funcs=normalize_funcs, normalize_funcs_training=normalize_funcs_training)

    def _plt_predict_volume(self, output_path, normalize_funcs):
        y_norm_func_rev = normalize_funcs['y_series_normalize_func_reverse']
        y_raw = y_norm_func_rev(self.y_vars).rename(columns={0: 'y_raw'})
        y_predict = y_norm_func_rev(pd.DataFrame([self.predict_y], columns=y_raw.index, index=['y_predict']).T)

        data_merged = pd.merge(y_raw, y_predict, left_index=True, right_index=True).rename(columns={y_raw.columns[0]: 'y_raw', y_predict.columns[0]: 'y_predict'})
        data_merged['ymd'] = list(map(lambda x: (x.year, x.month, x.day), data_merged.index))
        for key, data_one_day in data_merged.groupby('ymd'):
            fig = plt.figure()
            plt.plot(data_one_day['y_raw'].values, 'r-')
            plt.plot(data_one_day['y_predict'].values, 'b-')
            fig.savefig(output_path + 'predict_volume_vs_raw_volume' + '-'.join([str(k_) for k_ in key]) + '.jpg')
            plt.close()
        error_this_month = data_merged['y_raw'] - data_merged['y_predict']
        plt.hist(error_this_month.values, 100, facecolor='b')
        plt.axvline(0, color='red')
        plt.savefig(output_path + 'error_hist.jpg')
        plt.close()

        err_des = error_this_month.describe()
        err_des['skew'] = error_this_month.skew()
        err_des['kurt'] = error_this_month.kurt()

        err_des.to_csv(output_path + 'err_description.csv')

    def _daily_rsquared_output(self, output_path, normalize_funcs, normalize_funcs_training):
        y_norm_func_rev = normalize_funcs['y_series_normalize_func_reverse']
        y_norm_func_rev_training = normalize_funcs_training['y_series_normalize_func_reverse']
        y_raw = y_norm_func_rev(self.y_vars).rename(columns={0: 'y_raw'})
        y_predict = y_norm_func_rev(pd.DataFrame([self.predict_y], columns=y_raw.index, index=['y_predict']).T)

        data_merged = pd.merge(y_raw, y_predict, left_index=True, right_index=True).rename(columns={y_raw.columns[0]: 'y_raw', y_predict.columns[0]: 'y_predict'})
        data_merged['ymd'] = list(map(lambda x: (x.year, x.month, x.day), data_merged.index))

        data_merged['error'] = data_merged['y_raw'] - data_merged['y_predict']
        data_merged['sse'] = data_merged['y_raw'] - y_norm_func_rev_training(pd.DataFrame(self.model.endog)).mean().values

        def _generate_one_day_stats(c):
            mse_ = (c['sse'] * c['sse']).sum()
            msr_ = (c['error'] * c['error']).sum()
            r_sq_ = 1 - msr_ / mse_
            ret_ = pd.DataFrame([mse_, msr_, r_sq_], index=['mse', 'msr', 'rsquared']).T
            return ret_

        r_squared_daily = data_merged.groupby('ymd').apply(_generate_one_day_stats).unstack()
        r_squared_daily.to_csv(output_path + 'daily_rsquared.csv')
