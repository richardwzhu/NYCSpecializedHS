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

import matplotlib.pyplot as plt
plt.title("Correlation between Applications and Acceptances to HSPHS")
plt.xlabel("Applications")
plt.ylabel("Acceptances")
plt.plot(application, admission, 'o', color='black', markersize=2)
plt.plot(np.unique(application), np.poly1d(np.polyfit(application, admission, 1))(np.unique(application)))


# Correlation between application rate and admission
schoolSize = data[:,20]
appMod = application
admMod = admission
delete = []
for ii in range(len(schoolSize)):
    if schoolSize[ii] == "":
        delete.append(ii)
for ii in range(len(delete)):
    index = len(delete) - 1 - ii
    schoolSize = np.delete(schoolSize, delete[index])
    appMod = np.delete(appMod, delete[index])
    admMod = np.delete(admMod, delete[index])
schoolSize = schoolSize.astype(np.int)
appRate = appMod/schoolSize
corrArr = np.corrcoef(appRate, admMod)
corr = corrArr[0, 1]

# Calculating per student odds
bestOdds = 0
index = 0
for ii in range(len(schoolSize)):
    if schoolSize[ii] != 0:
        if admMod[ii]/schoolSize[ii] > bestOdds:
            bestOdds = admMod[ii]/schoolSize[ii]
            index = ii
dbn = data[:,0]
for ii in range(len(delete)):
    ind = len(delete) - 1 - ii
    dbn = np.delete(dbn, delete[ind])
print(dbn[index])

plt.title("Per Student Odds of Admission to HSPHS")
plt.xlabel("School Size")
plt.ylabel("Acceptances")
plt.plot(schoolSize, admMod, 'o', color='black', markersize=2)
