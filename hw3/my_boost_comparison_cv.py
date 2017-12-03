# Python3

import numpy as np
import pickle
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

tRange = [100,200,500,1000]
rho = 2**(-10)

score = {}
for T in tRange:
	# modified boost
	_myBoost = myBoost(base_estimator=baseStump,
		learning_rate=1.,
		n_estimators=T,
		algorithm="SAMME",
		rho=rho)
	_myBoost.fit(xTrain,yTrain)
	tmp = cross_val_score(_myBoost, xTrain, yTrain, cv=10)
	score['m',T] = 1. - tmp.mean()
	score['m',T,'std'] = 1. - tmp.std()
	print('modified AdaBoost -> ', score['m',T], ' +- ', score['m',T,'std'])
	# classic boost
	_skBoost = myBoost(base_estimator=baseStump,
		learning_rate=1.,
		n_estimators=T,
		algorithm="SAMME")
	_skBoost.fit(xTrain,yTrain)
	tmp = cross_val_score(_skBoost, xTrain, yTrain, cv=10)
	score['c',T] = 1. - tmp.mean()
	score['c',T,'std'] = 1. - tmp.std()
	print('classic AdaBoost -> ', score['c',T], ' +- ', score['c',T,'std'])
	
# save results

_list = [score,]
with open('data/cv_comp_data', 'wb') as _file:
    pickle.dump(_list, _file)
