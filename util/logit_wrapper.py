# -*- coding: utf-8 -*-
"""
Created on 2016/10/6 0:00

@author: qiding
"""

import numpy as np
import pandas as pd
import statsmodels.api


class LogitWrapper(statsmodels.api.Logit):
    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        params = statsmodels.api.Logit.fit(self)
        y_predict = statsmodels.api.Logit.predict(self, params=params, exog=self.exog)
        seperator = self._find_separator(y_predict=y_predict, y_raw=self.endog)
        return params, seperator

    def predict(self, params, exog=None, linear=False, separator=0):
        y = statsmodels.api.Logit.predict(self, params=params, exog=exog)
        y_ = np.where(y > separator, [1] * len(y), [0] * len(y))
        return y_

    @classmethod
    def _find_separator(self, y_predict, y_raw):
        df = pd.concat([y_predict, y_raw], axis=1, keys=['predict', 'raw'])
        df2 = df.sort_values('predict')
        df2['cumsum'] = df2['raw'].cumsum()
        idx_ = df2['cumsum'].idxmax()
        separator = df2.loc[idx_, 'predict']
        return separator
