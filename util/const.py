# -*- coding: utf-8 -*-
"""
Created on 2016/8/30 14:29

@author: qiding
"""


import datetime


MARKET_OPEN_TIME = datetime.datetime(1900, 1, 1, 9, 30, 0)
MARKET_END_TIME = datetime.datetime(1900, 1, 1, 15, 0, 0)
MARKET_OPEN_TIME_NOON = datetime.datetime(1900, 1, 1, 13, 0, 0)
MARKET_CLOSE_TIME_NOON = datetime.datetime(1900, 1, 1, 11, 30, 0)

FEE_RATE_UNILATERAL = .0008
DAYS_ONE_YEAR = 252

MIN_TICK_SIZE = 100

# TIME_SCALE_LIST = ['1min', '5min', '10min', '15min']
TIME_SCALE_LIST = ['15s', '1min', '5min', '10min', '15min']


class fitting_method:
    @property
    def OLS(self):
        return 'OLS'

    @property
    def DECTREE(self):
        return 'DecisionTreeRegression'

    @property
    def ADABOOST(self):
        return 'DecisionTreeAdaboost'

FITTING_METHOD = fitting_method()
