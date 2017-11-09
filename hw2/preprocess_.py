# Python3

import numpy as np
import sklearn.preprocessing
import sklearn.svm
import matplotlib.pyplot as plt

# write the data from 'spam.data' into an np array

tmp = []
with open('spam.data', 'r') as data_file:
    tmp.append([line.strip().split(",") for line in data_file])
data = np.array(tmp)[0,:,::-1].astype(np.float)

# get the Training and Test set

nTrain = 3000     
nTest = 1601

xTrain = data[:nTrain,1:]
yTrain = data[:nTrain,0]
xTest = data[-nTest:,1:]
yTest = data[-nTest:,0]

nFeat = xTrain.shape[1]

# scale the data

scaler = sklearn.preprocessing.StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)

# compute a random split of the training data

nBatch = 10
nTestBatch = np.int(nTrain / nBatch)
nTrainBatch = nTestBatch * (nBatch - 1)
tmp = np.arange(nTrain)
np.random.shuffle(tmp)
index = [list(tmp[i*nTestBatch:(i+1)*nTestBatch]) for i in range(nBatch)]

dTrainCV = []
dTestCV = np.zeros((nBatch,nTestBatch,nFeat))
lTrainCV = []
lTestCV = np.zeros((nBatch,nTestBatch))
for i in range(nBatch):
	# test batches
	for j in range(nTestBatch):
		dTestCV[i,j,:] = xTrainScaled[index[i][j],:]
		lTestCV[i,j] = yTrain[index[i][j]]
	# train batches
	d_tmp = []
	l_tmp = []
	for j in range(nTrain):
		if j not in index[i]:
			d_tmp.append(xTrain[j,:])
			l_tmp.append(yTrain[j])
	dTrainCV.append(d_tmp[:nTrainBatch])
	lTrainCV.append(l_tmp[:nTrainBatch])
dTrainCV = np.array(dTrainCV)
lTrainCV = np.array(lTrainCV)
	
print(dTrainCV.shape)
print(lTrainCV.shape)
print(dTestCV.shape)
print(lTestCV.shape)

# find (C,d)

k = 7
deg = [1]#,2,3,4]
C_range = [2**7]#2**i for i in range(2*k +1)]

cvScore = {}
for C in C_range:
	for d in deg:
		score = []
		for i in range(nBatch):
			svm = sklearn.svm.SVC(C=C, kernel='poly', degree=d)
			svm.fit(dTrainCV[i,:,:],lTrainCV[i,:])
			score.append(1.-svm.score(dTestCV[i,:,:],lTestCV[i,:]))
		score = np.array(score)
		cvScore[(d,C,'mean')] = np.mean(score)
		cvScore[(d,C,'std')] = np.std(score)
		print('C = ',C,' d = ',d,' -> Done')

colors = {1:'r',2:'b',3:'g',4:'k'}
for d in deg:
	tmp = np.array([cvScore[(d,C,'mean')] for C in C_range])
	tmp2 = np.array([cvScore[(d,C,'std')] for C in C_range])
	tmp_0 = np.log2(np.array(C_range))
	plt.plot(tmp_0, tmp, colors[d], label='d = ' + str(d))
	print(tmp)
	print(tmp2)
	plt.plot(tmp_0, tmp + tmp2, colors[d]+'--')
	plt.plot(tmp_0, tmp - tmp2, colors[d]+'--')	
plt.legend()
plt.show()
