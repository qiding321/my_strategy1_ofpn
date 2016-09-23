# -*- coding: utf-8 -*-
"""
Created on 2016/9/16 16:43

@author: qiding
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

import util.util


class RegDataTraining:
    def __init__(self, x_vars, y_vars):
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.num_of_x_vars = len(x_vars.columns)
        self.x_var_names = x_vars.columns
        self.paras = None
        self.model = None

    def fit(self, add_const=True):
        if add_const:
            self.model = sm.OLS(self.y_vars, sm.add_constant(self.x_vars), hasconst=True)
        else:
            self.model = sm.OLS(self.y_vars, self.x_vars, hasconst=False)
        self.paras = self.model.fit()
        return self.paras.rsquared

    def vars_iteration_model_selection(self, vars_left):
        for drop_var in vars_left:
            vars_new = [var_ for var_ in vars_left if var_ != drop_var]
            x_vars = self.x_vars[vars_new]
            y_vars = self.y_vars
            reg_data_new = RegDataTraining(x_vars=x_vars, y_vars=y_vars)
            yield reg_data_new

    def result_record_str(self):
        coef = self.paras.params
        tvalues = self.paras.tvalues
        df = pd.concat([coef, tvalues], keys=['coef', 'tvalues'], axis=1).T
        to_ret = util.util.pandas2str(df, title='training_result')

        to_ret += '\nrsquared,{}'.format(self.paras.rsquared)

        return to_ret


class RegDataTest:
    def __init__(self, x_vars, y_vars):
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.model = None
        self.paras = None

    def add_model(self, model, paras):
        self.model = model
        self.paras = paras

    def predict(self, add_const=True):  # todo
        if add_const:
            data_predict = self.model.predict(exog=sm.add_constant(self.x_vars), params=self.paras.params)
        else:
            data_predict = self.model.predict(exog=self.x_vars, params=self.paras.params)
        ssr = (pd.DataFrame(data_predict) - pd.DataFrame(self.y_vars.values)).values
        sse = (pd.DataFrame(self.y_vars.values) - self.model.endog.mean()).values  # for y_mean_in_sample, new
        # sse = (pd.DataFrame(self.y_vars.values) - pd.DataFrame(self.y_vars.values).mean()).values  # for y_mean_out_of_sample, old
        rsquared_out_of_sample = 1 - (ssr * ssr).sum()/(sse * sse).sum()
        return rsquared_out_of_sample

    def get_err(self, add_const=True):

        y_ = self.y_vars.values

        if add_const:
            data_predict = self.model.predict(exog=sm.add_constant(self.x_vars), params=self.paras.params)
        else:
            data_predict = self.model.predict(exog=self.x_vars, params=self.paras.params)
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
            'variance_x_contribution': self.x_vars.var() * (self.paras.params * self.paras.params),
            'rsquared_out_of_sample': rsquared_out_of_sample,
            'rsquared_out_of_sample_by_oos_mean': rsquared_out_of_sample_by_oos_mean,
            'var_y': var_y,
            'var_y_predict': var_y_predict,
            'bias_squared': bias_squared,
            'bias_mean': bias_mean,
            'cov_y_y_predict_multiplied_by_minus_2': cov_y_y_predict_multiplied_by_minus_2,
        }

        return ret

    def result_record_str(self):
        err_dict = self.get_err()
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
        reg_data_new = RegDataTest(x_vars=x_vars, y_vars=y_vars)
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

        data_to_rcd.to_csv(output_path + file_name)
