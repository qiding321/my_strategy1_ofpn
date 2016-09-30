# -*- coding: utf-8 -*-
"""
Created on 2016/9/30 11:24

@author: qiding
"""

import numpy as np
import pandas as pd

size = [100, 10]
df = pd.DataFrame(np.random.normal(size=size))
df.rolling(window=10).apply()
