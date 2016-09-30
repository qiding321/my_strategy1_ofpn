# -*- coding: utf-8 -*-
"""
Created on 2016/9/29 12:03

@author: qiding
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def generate_sample(model='ar1'):
    n = 10000
    std_err = 0.2
    a = 0.8
    b = 0.5

    func = lambda x, err: a * x + b + err

    if model == 'ar1':
        err = np.random.normal(loc=0, scale=std_err, size=[n, 1])
        x = pd.Series([np.nan] * n)
        y = pd.Series([np.nan] * n)
        x[0] = 0
        for i in range(n - 1):
            y[i] = func(x[i], err[i])
            x[i + 1] = y[i]
        y[n - 1] = func(x[n - 1], err[n - 1])

        return x, y


def reg(x, y):
    x_new = sm.add_constant(x)
    model = sm.OLS(y, x_new)
    params = model.fit()
    y_predict = model.predict(exog=x_new, params=params.params)
    # print(params.rsquared)
    return y_predict


def cal_rsquared(y_raw, y_predict, y_insample_mean=None):
    if y_insample_mean is None:
        y_insample_mean = y_raw.mean()
    mse = np.mean((y_raw - y_insample_mean) * (y_raw - y_insample_mean))
    msr = np.mean((y_predict - y_raw) * (y_predict - y_raw))
    r_sq = 1 - msr / mse
    return r_sq


def get_sum(s, lag):
    s_new = []
    x = 0
    while True:
        if x + lag >= len(s):
            break
        else:
            s_new.append(s[x:x + lag].sum())
        x += lag
    return pd.Series(s_new)


def main():
    x, y = generate_sample()

    y_predict = reg(x, y)
    r_sq = cal_rsquared(y, y_predict)
    print(r_sq)

    lag = 5
    x_5 = get_sum(x, lag)
    y_5 = get_sum(y, lag)
    y_predict_5 = reg(x_5, y_5)
    r_sq_5 = cal_rsquared(y_5, y_predict_5)
    print(r_sq_5)


if __name__ == '__main__':
    main()
