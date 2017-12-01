# Python3

import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from weight_boosting_rho import AdaBoostClassifier as myBoost
from sklearn.ensemble import AdaBoostClassifier as skBoost

# load data

with open ('data/data_python', 'rb') as _file:
    [xTrain,yTrain,xTest,yTest] = pickle.load(_file)

# boost

baseStump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
baseStump.fit(xTrain, yTrain)

# modified boost with best parameters - test error

T =
rho = 

_myBoost = myBoost(base_estimator=baseStump,
	learning_rate=1.,
	n_estimators=T,
	algorithm="SAMME",
	rho=rho)

_myBoost.fit(xTrain,yTrain)
print('modified AdaBoost -> ', _myBoost.score(xTest,yTest))

# classic boost with best parameters - test error

_skBoost = skBoost(base_estimator=baseStump,
	learning_rate=1.,
	n_estimators=T,
	algorithm="SAMME")

_skBoost.fit(xTrain,yTrain)
print('classic AdaBoost -> ', _skBoost.score(xTest,yTest))
