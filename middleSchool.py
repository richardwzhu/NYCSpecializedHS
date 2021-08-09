#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:32:56 2021

@author: richardzhu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
import scipy.stats as st

# Remove commas from school names
with open("middleSchoolData.csv") as infile, open("fixedData.csv", "w") as outfile:
    for line in infile:
        outfile.write(line.replace(", ", " "))
    
# Retrieve clean data
data = np.genfromtxt('fixedData.csv', delimiter=',', skip_header = 1, dtype='unicode')

# Correlation between application and admission
application = data[:,2].astype(int)
admission = data[:,3].astype(int)
corrArr = np.corrcoef(application, admission)
corr = corrArr[0, 1]

plt.figure(0)
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
schoolSize = schoolSize.astype(int)
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

plt.figure(1)
plt.title("Per Student Odds of Admission to HSPHS")
plt.xlabel("School Size")
plt.ylabel("Acceptances")
plt.plot(schoolSize, admMod, 'o', color='black', markersize=2)

#Multiple regression
X = np.transpose([data[:,11],data[:,12],data[:,13],data[:,14],data[:,15],data[:,16]])

A = data[:,21] # student achievement
R = data[:,22] # reading exceedd
M = data[:,23] # math exceed

delete = []
for ii in range(len(X)):
    for jj in range(len(X[0])):
        if X[ii,jj] == "":
            if ii not in delete: delete.append(ii)
    if A[ii] == "" or R[ii] == "" or M[ii] == "":
        if ii not in delete: delete.append(ii)
for ii in range(len(delete)):
    index = len(delete) - 1 - ii
    X = np.delete(X, delete[index], axis=0)
    A = np.delete(A, delete[index])
    R = np.delete(R, delete[index])
    M = np.delete(M, delete[index])
    
X = X.astype(float)
A = A.astype(float)
R = R.astype(float)
M = M.astype(float)

regr = linear_model.LinearRegression() 
regr.fit(X,A)
rSqrA = regr.score(X,A)
regr.fit(X,R)
rSqrR = regr.score(X,R)
regr.fit(X,M)
rSqrM = regr.score(X,M)

plt.figure(2)
fig, axs = plt.subplots(2, 3)
fig.suptitle('Relationships Between Student Perception and Student Achievement')
axs[0, 0].plot(X[:,0], A, 'o', markersize=0.75)
axs[0, 0].set_title('Rigorous instruction', fontsize=7)
axs[0, 1].plot(X[:,1], A, 'o', color='orange', markersize=0.75)
axs[0, 1].set_title('Collaborative teachers', fontsize=7)
axs[0, 2].plot(X[:,2], A, 'o', color='green', markersize=0.75)
axs[0, 2].set_title('Supportive environment', fontsize=7)
axs[1, 0].plot(X[:,3], A, 'o', color='red', markersize=0.75)
axs[1, 0].set_title('Effictive school leadership', fontsize=7)
axs[1, 1].plot(X[:,4], A, 'o', color='purple', markersize=0.75)
axs[1, 1].set_title('Strong family community ties', fontsize=7)
axs[1, 2].plot(X[:,5], A, 'o', color='black', markersize=0.75)
axs[1, 2].set_title('Trust', fontsize=7)
for ii in range(2):
    for jj in range(3):
        axs[ii, jj].get_xaxis().set_visible(False)
        axs[ii, jj].get_yaxis().set_visible(False)

plt.figure(3)
fig, axs = plt.subplots(2, 3)
fig.suptitle('Relationships Between Student Perception and Reading Scores')
axs[0, 0].plot(X[:,0], R, 'o', markersize=0.75)
axs[0, 0].set_title('Rigorous instruction', fontsize=7)
axs[0, 1].plot(X[:,1], R, 'o', color='orange', markersize=0.75)
axs[0, 1].set_title('Collaborative teachers', fontsize=7)
axs[0, 2].plot(X[:,2], R, 'o', color='green', markersize=0.75)
axs[0, 2].set_title('Supportive environment', fontsize=7)
axs[1, 0].plot(X[:,3], R, 'o', color='red', markersize=0.75)
axs[1, 0].set_title('Effictive school leadership', fontsize=7)
axs[1, 1].plot(X[:,4], R, 'o', color='purple', markersize=0.75)
axs[1, 1].set_title('Strong family community ties', fontsize=7)
axs[1, 2].plot(X[:,5], R, 'o', color='black', markersize=0.75)
axs[1, 2].set_title('Trust', fontsize=7)
for ii in range(2):
    for jj in range(3):
        axs[ii, jj].get_xaxis().set_visible(False)
        axs[ii, jj].get_yaxis().set_visible(False)

# Small vs large school
schoolSizeMod = schoolSize[appMod>0]
admMod = admMod[appMod>0]
appMod = appMod[appMod>0]
admRate = admMod/appMod
median = np.median(schoolSizeMod)
smallSchools = admRate[schoolSizeMod < median]
largeSchools = admRate[schoolSizeMod >= median]
pValSL = st.ttest_ind(smallSchools, largeSchools)
