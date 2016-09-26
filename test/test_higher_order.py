# -*- coding: utf-8 -*-
"""
Created on 2016/9/26 10:09

@author: qiding
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

n = 10000
std_e = .1

a = 1.6
b = 2.2

x = np.random.random(n)
xx = x * x
e = np.random.random(n) * std_e

y = xx * a + b + e

reg_var = x

reg_res = sm.OLS(endog=y, exog=sm.add_constant(reg_var)).fit()

y_predict = (reg_res.params * sm.add_constant(reg_var)).sum(1)
residual = y - y_predict

print(reg_res.rsquared)
print(np.corrcoef(residual, x))

plt.plot(x, y)
