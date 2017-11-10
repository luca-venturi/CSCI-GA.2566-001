# use Python3

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# load data

with open ('data/data_python', 'rb') as _file:
    [xTrain,yTrain,xTest,yTest] = pickle.load(_file)

# compute customized Gram matrix

def my_kernel(x,y): 
	n = 4
	c = 0.1 # = 2/(n(n+1))
	G = 0.
	p = np.dot(x,y.T)
	for i in range(n):
		for j in range(i,n):
			G += p ** (i+j+2)
	return G * c

gTrain = my_kernel(xTrain,xTrain)
print('Gram matrix computed')

# run cross validation

K = 7
C_range = [2**(i-K) for i in range(2*K +1)] # change? too slow
nBatch = 10

scores = {}
for C in C_range:
	svm = SVC(C=C, kernel='precomputed')
	scores[C] = cross_val_score(svm,gTrain,yTrain,cv=nBatch).mean()
	print('C = ',C,' -> ',scores[C])

# find best C

errors = np.array([(1.-scores[C]) for C in C_range])
print(errors)
best_C = C_range[np.argmin(errors)]
print(best_C)

print('Cross validation error: ', errors[best_C])

# test error

svm = SVC(C=best_C, kernel='precomputed')
svm.fit(gTrain,yTrain)
print('Test error: ', 1. - svm.score(xTest,yTest))

