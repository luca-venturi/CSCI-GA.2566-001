# Python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from weight_boosting_rho import AdaBoostClassifier as myBoost
from sklearn.ensemble import AdaBoostClassifier as skBoost
from sklearn.model_selection import cross_val_score

# load data

with open ('data/data_python', 'rb') as _file:
    [xTrain,yTrain,xTest,yTest] = pickle.load(_file)

# boost

baseStump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
baseStump.fit(xTrain, yTrain)

# modified boost with best parameters - margin

T = 500
rho = 2**(-10)

_myBoost = myBoost(base_estimator=baseStump,
	learning_rate=1.,
	n_estimators=T,
	algorithm="SAMME",
	rho=rho)

_myBoost.fit(xTrain,yTrain)
myMargin = _myBoost.margin(xTrain,yTrain)

# classic boost with best parameters - margin

_skBoost = myBoost(base_estimator=baseStump,
	learning_rate=1.,
	n_estimators=T,
	algorithm="SAMME")

_skBoost.fit(xTrain,yTrain)
skMargin = _skBoost.margin(xTrain,yTrain)

# plot

plt.title('Cumulative margin ($T=500$)')
plt.xlabel('$\\theta$')
plt.ylabel('margin')

plt.plot(myMargin[0,:],myMargin[1,:],label='mAdaBoost')
plt.plot(skMargin[0,:],skMargin[1,:],label='AdaBoost')

plt.legend()
plt.savefig('margin', bbox_inches='tight')
plt.show() #
