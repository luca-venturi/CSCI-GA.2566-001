# Python3

import numpy as np
import sklearn.preprocessing as prep
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# write data from 'data/spam' into np.array

tmp = []
with open('data/spam', 'r') as data_file:
    tmp.append([line.strip().split(",") for line in data_file])
data = np.array(tmp)[0,:,:].astype(np.float)

# get Train and Test set

nTrain = 3000     
nTest = 1601

xTrain = data[:nTrain,:-1]
yTrain = data[:nTrain,-1].astype(np.int)
xTest = data[-nTest:,:-1]
yTest = data[-nTest:,-1].astype(np.int)

nFeat = xTrain.shape[1]

# scale data

scaler = prep.StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# run cross validation

K = 7
degree_range = [1,2,3,4]
C_range = [2**(i-K) for i in range(2*K +1)]
nBatch = 10

scores = {}
for d in degree_range:
	for C in C_range:
		svm = SVC(C=C, kernel='poly', degree = d)
		tmp = cross_val_score(svm,xTrain,yTrain,cv = nBatch)
		scores[d,C,'mean'] = tmp.mean()
		scores[d,C,'std'] = tmp.std()
		print('d = ',d,' C = ',C,' -> ',scores[d,C,'mean'])
		
# plot CV errors

colors = {1:'r',2:'b',3:'g',4:'k'}
errors = np.array([[(1.-scores[d,C,'mean']) for C in C_range] for d in degree_range])
std = np.array([[scores[d,C,'std'] for C in C_range] for d in degree_range])
print(errors)
best_index = np.unravel_index(np.argmin(errors), errors.shape)

log2_C_range = np.log2(np.array(C_range))
for i in range(len(degree_range)):
	plt.plot(log2_C_range,errors[i,:],colors[degree_range[i]], label='d = ' + str(degree_range[i]))
	#plt.plot(log2_C_range,errors[i,:] + std[i,:],colors[degree_range[i]]+'--')
	#plt.plot(log2_C_range,errors[i,:] - std[i,:],colors[degree_range[i]]+'--')
	#plt.errorbar(log2_C_range, errors[i,:], xerr = None, yerr=std[i,:], colors[degree_range[i]], label='d = ' + str(degree_range[i]))
plt.legend()
plt.show()

# find best (d,C)

best_d = degree_range[best_index[0]]
best_C = C_range[best_index[1]]
print(d,C)

# plot CV errors

scores = {}
for d in degree_range:
	svm = SVC(C=best_C, kernel='poly', degree = d)
	tmp = cross_val_score(svm,xTrain,yTrain,cv = nBatch)
	scores[d,'cv'] = tmp.mean()
	print('d = ',d,' C = ',best_C,' -> ',scores[d,'cv'])
	
errors_cv = np.array([(1.-scores[d,'cv']) for d in degree_range])
plt.plot(degree_range,errors_cv,'b')

# plot test errors

for d in degree_range:
	svm = SVC(C=best_C, kernel='poly', degree = d)
	svm.fit(xTrain,yTrain)
	scores[d,'n_support'] = sum(svm.n_support_)
	scores[d,'test'] = svm.score(xTest,yTest)
	print('d = ',d,' C = ',best_C,' -> ',scores[d,'test'])
	
errors_test = np.array([(1.-scores[d,'test']) for d in degree_range])
plt.plot(degree_range,errors_test,'y')
plt.show()

# plot support vectors

n_support = np.array([scores[d,'n_support'] for d in degree_range])
plt.plot(degree_range,n_support,'r')
plt.show()

# how to evaluate number of support vectors on margin hyperplanes

# customized kernel

def my_kernel(x,y):
	c = 0.1
	n = 4
	g = 0.
	p = np.dot(x,y.T)
	v = [1,1,2,2,2,1,1]
	for i in range(n+3):
		g += v[i] * (p ** (i+2))
	return g
Gram = my_kernel(xTrain,xTrain)

# run cross validation

K = 7
degree_range = [1,2,3,4]
C_range = [2**(i-K) for i in range(2*K +1)]
nBatch = 10

scores = {}
for C in C_range:
	svm = SVC(C=C, kernel='precomputed')
	tmp = cross_val_score(svm,Gram,yTrain,cv = nBatch)
	scores[C,'mean'] = tmp.mean()
	scores[C,'std'] = tmp.std()
	print(' C = ',C,' -> ',scores[C,'mean'])
		
# plot CV errors

colors = {1:'r',2:'b',3:'g',4:'k'}
errors = np.array([(1.-scores[C,'mean']) for C in C_range])
std = np.array([scores[C,'std'] for C in C_range])
print(errors)

log2_C_range = np.log2(np.array(C_range))
plt.plot(log2_C_range,errors,'r-', label='d = ' + str(degree_range[i]))
	#plt.plot(log2_C_range,errors[i,:] + std[i,:],colors[degree_range[i]]+'--')
	#plt.plot(log2_C_range,errors[i,:] - std[i,:],colors[degree_range[i]]+'--')
	#plt.errorbar(log2_C_range, errors[i,:], xerr = None, yerr=std[i,:], colors[degree_range[i]], label='d = ' + str(degree_range[i]))
plt.legend()
plt.show()

# find best C

best_index = np.argmin(errors)
best_C = C_range[best_index]
print(C)

# print CV errors

scores = {}
svm = SVC(C=best_C, kernel='precomputed')
tmp = cross_val_score(svm,Gram,yTrain,cv = nBatch)
scores['best_C','cv'] = tmp.mean()
print(' C = ',best_C,' -> ',scores['best_C','cv'])
	
print(1.-scores['best_C','cv'])

# print test errors

svm = SVC(C=best_C, kernel='precomputed')
svm.fit(Gram,yTrain)
scores['best_C','test'] = svm.score(xTest,yTest)
	
print(1.-scores['best_C','test'])
