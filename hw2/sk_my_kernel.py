# use Python3

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# load data
# does not work with (-1.,1.) scaling. works with (-0.1,0.1) scaling, needs larger C though
# --> re-preprocess

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

K = 15
C_range = [2**(i+1) for i in range(K)]
nBatch = 10

scores = {}
for C in C_range:
	svm = SVC(C=C, kernel='precomputed')
	scores[C] = cross_val_score(svm,gTrain,yTrain,cv=nBatch).mean()
	print('C = ',C,' -> ',scores[C])

# find best C

errors = np.array([(1.-scores[C]) for C in C_range])
best_C = C_range[np.argmin(errors)]
print('C* = ',best_C)

print('Cross validation error: ', 1.-scores[best_C])

# test error

svm = SVC(C=best_C, kernel='precomputed')
svm.fit(gTrain,yTrain)
print('Test error: ', 1. - svm.score(my_kernel(xTest,xTrain),yTest))
