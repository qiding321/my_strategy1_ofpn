# -*- coding: utf-8 -*-
"""
Created on 2016/8/30 10:36

@author: qiding
"""

import logging
import os

from my_path.path import log_path_root


class Logger(logging.Logger):
    def __init__(self, name='log', level=logging.INFO, log_path2=None):
        logging.Logger.__init__(self, name, level)
        self.log_path = log_path_root + name + '.log'
        if not os.path.exists(log_path_root):
            os.makedirs(log_path_root)
        self.formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        file_log = logging.FileHandler(self.log_path)
        file_log.setFormatter(self.formatter)
        self.addHandler(file_log)

        if log_path2 is not None:
            log_path2_root, file_name_ = os.path.split(log_path2)
            if not os.path.exists(log_path2_root):
                os.makedirs(log_path2_root)
            file_log2 = logging.FileHandler(log_path2)
            file_log2.setFormatter(self.formatter)
            self.addHandler(file_log2)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(self.formatter)
        self.addHandler(console)

    def add_path(self, log_path2):
        log_path2_root, file_name_ = os.path.split(log_path2)
        if not os.path.exists(log_path2_root):
            os.makedirs(log_path2_root)
        file_log2 = logging.FileHandler(log_path2)
        file_log2.setFormatter(self.formatter)
        self.addHandler(file_log2)


    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

log_base = Logger(name='MarketMakingStrategyLog')

log_price_predict = Logger(name='MM_Price_Predict_Log')

log_order_flow_predict = Logger(name='MM_Order_Flow_Predict')

log_error = Logger(name='Error')