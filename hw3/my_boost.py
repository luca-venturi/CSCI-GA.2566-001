# Python3

import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from weight_boosting_rho import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# load data

with open ('data/data_python', 'rb') as _file:
    [xTrain,yTrain,xTest,yTest] = pickle.load(_file)

# boost

baseStump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
baseStump.fit(xTrain, yTrain)

tRange = [100,200,500,1000]
rhoRange = [2**(-i-1) for i in range(10)]

score = {}
for T in tRange:
	for rho in rhoRange:
		boostClassifier = AdaBoostClassifier(base_estimator=baseStump,
			learning_rate=1.,
			n_estimators=T,
			algorithm="SAMME",
			rho=rho)
		score[rho,T] = 1. - cross_val_score(boostClassifier, xTrain, yTrain, cv=10).mean()
		print( 'rho = ', rho, ' T = ', T, ' -> ', score[rho,T])
best_rho, best_T = min(score, key=score.get)
print(best_rho, best_T)
