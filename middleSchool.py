#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:32:56 2021

@author: richardzhu
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = np.genfromtxt('middleSchoolData.csv', delimiter = ',', skip_header = 1)