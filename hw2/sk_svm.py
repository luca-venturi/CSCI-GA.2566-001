# Python3

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# load data

with open ('data/data_python', 'rb') as _file:
    [xTrain,yTrain,xTest,yTest] = pickle.load(_file)

# run cross validation

K = 7
degree_range = [1,2,3,4]
C_range = [2**(i-K) for i in range(2*K +1)]
nBatch = 10

scores = {}
print('Cross validation scores:')
for d in degree_range:
	for C in C_range:
		svm = SVC(C=C, kernel='poly', degree = d)
		tmp = cross_val_score(svm,xTrain,yTrain,cv = nBatch)
		scores[d,C,'mean'] = tmp.mean()
		scores[d,C,'std'] = tmp.std()
		print('d = ',d,' C = ',C,' -> ',scores[d,C,'mean'])
		
# finds best (d,C)

colors = {1:'r',2:'b',3:'g',4:'k'}
errors = np.array([[(1.-scores[d,C,'mean']) for C in C_range] for d in degree_range])
std = np.array([[scores[d,C,'std'] for C in C_range] for d in degree_range])
best_index = np.unravel_index(np.argmin(errors), errors.shape)
best_d = degree_range[best_index[0]]
best_C = C_range[best_index[1]]
print('d* = ',best_d,' C* = ',best_C)

# plot CV errors

log2_C_range = np.log2(np.array(C_range))
for i in range(len(degree_range)):
	plt.plot(log2_C_range,errors[i,:],colors[degree_range[i]], label='d = ' + str(degree_range[i]))
	#plt.plot(log2_C_range,errors[i,:] + std[i,:],colors[degree_range[i]]+'--')
	#plt.plot(log2_C_range,errors[i,:] - std[i,:],colors[degree_range[i]]+'--')
	#plt.errorbar(log2_C_range, errors[i,:], xerr = None, yerr=std[i,:], colors[degree_range[i]], label='d = ' + str(degree_range[i]))
plt.legend()
plt.show()

# cross_validation and test for best_C

b_scores = {}
for d in degree_range:
	b_scores[d,'cv'] = scores[d,best_C,'mean']
errors_cv = np.array([(1.-b_scores[d,'cv']) for d in degree_range])
plt.plot(degree_range,errors_cv,'b')

print('Test scores:')
for d in degree_range:
	svm = SVC(C=best_C, kernel='poly', degree = d)
	svm.fit(xTrain,yTrain)
	b_scores[d,'n_support'] = sum(svm.n_support_)
	b_scores[d,'test'] = svm.score(xTest,yTest)
	print('d = ',d,' C = ',best_C,' -> ',scores[d,'test'])
	
errors_test = np.array([(1.-b_scores[d,'test']) for d in degree_range])
plt.plot(degree_range,errors_test,'y')
plt.show()

# plot support vectors

n_support = np.array([b_scores[d,'n_support'] for d in degree_range])
plt.plot(degree_range,n_support,'r')
plt.show()

# how to evaluate number of support vectors on margin hyperplanes??

