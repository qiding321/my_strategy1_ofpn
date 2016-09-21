# -*- coding: utf-8 -*-
"""
Created on 2016/8/30 10:36

@author: qiding
"""


import logging
import os
from my_path.path import log_path_root


class Logger(logging.Logger):
    def __init__(self, name='log', level=logging.INFO):
        logging.Logger.__init__(self, name, level)
        self.log_path = log_path_root + name + '.log'
        if not os.path.exists(log_path_root):
            os.makedirs(log_path_root)
        file_log = logging.FileHandler(self.log_path)

        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        file_log.setFormatter(formatter)
        self.addHandler(file_log)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        self.addHandler(console)

    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

log_base = Logger(name='MarketMakingStrategyLog')

log_price_predict = Logger(name='MM_Price_Predict_Log')

log_order_flow_predict = Logger(name='MM_Order_Flow_Predict')

log_error = Logger(name='Error')