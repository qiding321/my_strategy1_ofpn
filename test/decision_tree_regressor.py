# -*- coding: utf-8 -*-
"""
Created on 2016/9/28 10:19

@author: qiding
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def generate_random_normal(size, mu, std):
    d = np.random.normal(loc=mu, scale=std, size=size)
    return pd.DataFrame(d)


def generate_sample(func, length):
    mu = 0
    std = 1

    mu_err = 0
    std_err = 0.1

    x = generate_random_normal([length, 2], mu=mu, std=std)
    err = generate_random_normal(length, mu=mu_err, std=std_err)[0]
    y = func(x, err)
    return y, x


def cal_r_squared(y_raw, y_predict, y_training):
    e = y_raw - y_training.mean()
    mse = (e * e).mean()
    r = y_raw - y_predict
    msr = (r * r).mean()
    r_sq = 1 - msr / mse
    return r_sq


def main():
    # transform function
    func = lambda x, err: x[0] * x[0] + x[1] * x[1] + err

    # data generation
    training_length = 10000
    training_y, training_x = generate_sample(func=func, length=training_length)

    testing_length = 1000
    testing_y, testing_x = generate_sample(func=func, length=testing_length)

    # training and fitting
    training_model = DecisionTreeRegressor(max_depth=10)
    training_model.fit(training_x, training_y)
    y_predict = training_model.predict(training_x)
    y_predict_oos = training_model.predict(testing_x)

    # r-squared
    r_sq_in_sample = cal_r_squared(y_raw=training_y, y_predict=y_predict, y_training=training_y)
    r_sq_out_of_sample = cal_r_squared(y_raw=testing_y, y_predict=y_predict_oos, y_training=training_y)
    print('rsq_in_sample: {}\nrsq_out_of_sample: {}'.format(r_sq_in_sample, r_sq_out_of_sample))


if __name__ == '__main__':
    main()
