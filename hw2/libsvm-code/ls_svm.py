from svmutil import *

# load data

yTrain, xTrain = svm_read_problem('../../data/train_scaled')
p = svm_problem(yTrain, xTrain)

# cross-validation

c = 2 ** 8
d_range = [1,2,3,4]

nSV = {}
nSVm = {}
for d in d_range:
	par = svm_parameter('-t 1'+' -d '+str(d)+' -c '+str(c)+' -h 0')
	m = svm_train(p, par)
