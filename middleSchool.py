#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:32:56 2021

@author: richardzhu
"""

import numpy as np

# Remove commas from school names
with open("middleSchoolData.csv") as infile, open("fixedData.csv", "w") as outfile:
    for line in infile:
        outfile.write(line.replace(", ", " "))
    
# Retrieve clean data
data = np.genfromtxt('fixedData.csv', delimiter=',', skip_header = 1, dtype='unicode')

# Correlation between application and admission
application = data[:,2].astype(np.int)
admission = data[:,3].astype(np.int)
corrArr = np.corrcoef(application, admission)
corr = corrArr[0, 1]

