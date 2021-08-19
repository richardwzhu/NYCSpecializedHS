#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:32:56 2021

@author: richardzhu
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.decomposition import PCA

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
plt.title("Correlation Between Applications and Acceptances to HSPHS")
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

plt.figure(1)
plt.title("Correlation Between Application Rate and Acceptances to HSPHS")
plt.xlabel("Application Rate")
plt.ylabel("Acceptances")
plt.plot(appRate, admMod, 'o', color='black', markersize=2)
plt.plot(np.unique(appRate), np.poly1d(np.polyfit(appRate, admMod, 1))(np.unique(appRate)))

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

plt.figure(2)
plt.title("Per Student Odds of Admission to HSPHS")
plt.xlabel("School Size")
plt.ylabel("Acceptances")
plt.plot(schoolSize, admMod, 'o', color='black', markersize=2)

#PCA and linear regression
perception = np.transpose([data[:,11],data[:,12],data[:,13],data[:,14],data[:,15],data[:,16]])
achievement =  np.transpose([data[:,21],data[:,22],data[:,23]])

delete = []
for ii in range(len(perception)):
    for jj in range(len(perception[0])):
        if perception[ii,jj] == "":
            if ii not in delete: delete.append(ii)
    for jj in range(len(achievement[0])):
        if achievement[ii,jj] == "":
            if ii not in delete: delete.append(ii)
for ii in range(len(delete)):
    index = len(delete) - 1 - ii
    perception = np.delete(perception, delete[index], axis=0)
    achievement = np.delete(achievement, delete[index], axis=0)
    
perception = perception.astype(float)
achievement = achievement.astype(float)

plt.figure(3)
rP = np.corrcoef(perception,rowvar=False)
plt.imshow(rP) 
plt.colorbar()

plt.figure(4)
rA = np.corrcoef(achievement,rowvar=False)
plt.imshow(rA) 
plt.colorbar()

zscoredDataP = st.zscore(perception)
pcaP = PCA().fit(zscoredDataP)
eigValsP = pcaP.explained_variance_
loadingsP = pcaP.components_
newP = PCA().fit_transform(zscoredDataP)
numPerceptions = 6

plt.figure(5)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.bar(np.linspace(1,6,6),eigValsP)
plt.plot([0,numPerceptions],[1,1],color='red',linewidth=1) 

plt.figure(6)
plt.xlabel('Student Perception')
plt.ylabel('Loading')
plt.bar(np.linspace(1,6,6),loadingsP[0,:]*-1)

zscoredDataA = st.zscore(achievement)
pcaA = PCA().fit(zscoredDataA)
eigValsA = pcaA.explained_variance_
loadingsA = pcaA.components_
newA = PCA().fit_transform(zscoredDataA)
numMeasurements = 3

plt.figure(7)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.bar(np.linspace(1,3,3),eigValsA)
plt.plot([0,numMeasurements],[1,1],color='red',linewidth=1) 

plt.figure(8)
plt.xlabel('Objective Achievement')
plt.ylabel('Loading')
plt.bar(np.linspace(1,3,3),loadingsA[0,:]*-1)

perceptionComponent = newP[:,0]
achievementComponent = newA[:,0]
slope, intercept, r_value, p_value, std_err = st.linregress(perceptionComponent, achievementComponent)
cod = r_value**2

# Small vs large school
schoolSizeMod = schoolSize[appMod>0]
admMod = admMod[appMod>0]
appMod = appMod[appMod>0]
admRate = admMod/appMod
median = np.median(schoolSizeMod)
smallSchools = admRate[schoolSizeMod < median]
largeSchools = admRate[schoolSizeMod >= median]
pValSL = st.ttest_ind(smallSchools, largeSchools)

plt.title("Correlation Between School Size and Acceptance Rate to HSPHS")
plt.xlabel("School Size")
plt.ylabel("Acceptance Rate")
plt.scatter(schoolSizeMod[schoolSizeMod < median], smallSchools, s=1.5, c='b', marker="s", label='Small Schools')
plt.scatter(schoolSizeMod[schoolSizeMod >= median], largeSchools, s=1.5, c='r', marker="s", label='Large Schools')
plt.legend(loc='upper right');
plt.show()

#Per student spending vs student achievemnt
spending = data[:,4]
obj = np.transpose([data[:,21],data[:,22],data[:,23]])
delete = []
for ii in range(len(spending)):
    if spending[ii] == "":
        if ii not in delete: delete.append(ii)
    for jj in range(len(obj[0])):
        if obj[ii,jj] == "":
            if ii not in delete: delete.append(ii)
for ii in range(len(delete)):
    index = len(delete) - 1 - ii
    spending = np.delete(spending, delete[index])
    obj = np.delete(obj, delete[index], axis=0)
spending = spending.astype(int)
obj = obj.astype(float)

zs = st.zscore(obj)
pcaObj = PCA().fit(zs)
eigValues = pcaObj.explained_variance_ 
loadings = pcaObj.components_ *-1
newObj = pcaObj.fit_transform(zs)*-1                    

achievement = newObj[:,0]

plt.figure(9)
plt.title("Correlation Between Per Student Spending and Objective Student Achievement")
plt.xlabel("Per Student Spending")
plt.ylabel("Objective Student Achievement")
plt.plot(spending, achievement, 'o', color='black', markersize=2)
medianSpending = np.median(spending)
lowSpending = achievement[spending < medianSpending]
highSpending = achievement[spending >= medianSpending]
pValObj = st.ttest_ind(lowSpending, highSpending) 

# Proportion of schools
totalAcceptances = np.sum(admission)
sortedAcceptances = np.sort(admission)[::-1] #sort in descending order
total = 0
counter = 0

for ii in sortedAcceptances:
    total += ii
    counter += 1
    if total >= (totalAcceptances*0.9):
        break   
proportion = counter/len(admission)

plt.figure(11)
plt.title("Number of Acceptances Sorted in Decreasing Order")
plt.xlabel("Schools in Decreasing Order")
plt.ylabel("Number of Acceptances")
plt.bar(np.linspace(1,200,200),sortedAcceptances[0:200])
plt.axvline(x=123, color="red")

#Prediction
predictors = np.transpose([data[:,2],data[:,4],data[:,5],data[:,6],data[:,7],data[:,8],
                           data[:,9],data[:,10],data[:,11],data[:,12],data[:,13],data[:,14],
                           data[:,15],data[:,16],data[:,17],data[:,18],data[:,19],data[:,20]])
adm = admission
ach = np.transpose([data[:,21],data[:,22],data[:,23]])

delete = []
for ii in range(len(predictors)):
    for jj in range(len(predictors[0])):
        if predictors[ii,jj] == "" or adm[ii] == "":
            if ii not in delete: delete.append(ii)
    for jj in range(len(ach[0])):
        if ach[ii,jj] == "":
            if ii not in delete: delete.append(ii)
for ii in range(len(delete)):
    index = len(delete) - 1 - ii
    predictors = np.delete(predictors, delete[index], axis=0)
    adm = np.delete(adm, delete[index])
    ach = np.delete(ach, delete[index], axis=0)
    
predictors = predictors.astype(float)
ach = ach.astype(float)
    
zscoredData = st.zscore(predictors)
pca = PCA().fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_ *-1
origDataNewCoordinates = pca.fit_transform(zscoredData)*-1

plt.figure(12)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.bar(np.linspace(1,len(predictors[0]),len(predictors[0])),eigValues)
plt.plot([0,len(predictors[0])],[1,1],color='red',linewidth=1)

plt.figure(13)
plt.xlabel('Predictors')
plt.ylabel('Loading')
plt.bar(np.linspace(1,len(predictors[0]),len(predictors[0])),loadings[0,:])                   
plt.bar(np.linspace(1,len(predictors[0]),len(predictors[0])),loadings[1,:])

schoolDistrictConditions = origDataNewCoordinates[:,0]
relationalConditions = origDataNewCoordinates[:,1]
slope, intercept, r_value, p_value, std_err = st.linregress(schoolDistrictConditions, adm)
codSD = r_value**2
slope, intercept, r_value, p_value, std_err = st.linregress(relationalConditions, adm)
codRC = r_value**2    

zscoredAch = st.zscore(ach)
pcaAch = PCA().fit(zscoredAch)
eigValues = pcaAch.explained_variance_ 
loadings = pcaAch.components_ *-1
newAch = pcaAch.fit_transform(zscoredAch)*-1                    

objectiveAchievement = newAch[:,0]
slope, intercept, r_value, p_value, std_err = st.linregress(schoolDistrictConditions, objectiveAchievement)
codSDOA = r_value**2
slope, intercept, r_value, p_value, std_err = st.linregress(relationalConditions, objectiveAchievement)
codRCOA = r_value**2    
