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
nTrainBatch = np.int(nTrain / nBatch)
tmp = np.arange(nTrain)
np.random.shuffle(tmp)
index = [list(tmp[i*nTrainBatch:(i+1)*nTrainBatch]) for i in range(nBatch)]

dTrain = np.zeros((nBatch,nTrainBatch,nFeat))
lTrain = np.zeros((nBatch,nTrainBatch))
for i in range(nBatch):
	for j in range(nTrainBatch):
		dTrain[i,j,:] = xTrainScaled[index[i][j],:]
		lTrain[i,j] = yTrain[index[i][j]]
	
print(dTrain.shape)
print(lTrain.shape)

# find (C,d)

k = 5
deg = [4]#,2,3,4]
C_range = [2**i for i in range(2*k +1)] #**(i-k)

cvScore = {}
for C in C_range:
	for d in deg:
		score = []
		for i in range(nBatch):
			svm = sklearn.svm.SVC(C=C, kernel='poly', degree=d)
			svm.fit(dTrain[i,:,:],lTrain[i,:])
			x_tmp = []
			y_tmp = []
			for j in range(nTrain):
				if j not in index[i]:
					x_tmp.append(xTrainScaled[j,:])
					y_tmp.append(yTrain[j])
			x_tmp = np.array(x_tmp)
			y_tmp = np.array(y_tmp)
			score.append(1.-svm.score(x_tmp,y_tmp))
		score = np.array(score)
		cvScore[(d,C,'mean')] = np.mean(score)
		cvScore[(d,C,'std')] = np.std(score)
		print('C = ',C,' d = ',d,' -> Done')
		
for d in deg:
	tmp = np.array([cvScore[(d,C,'mean')] for C in C_range])
	tmp2 = np.array([cvScore[(d,C,'std')] for C in C_range])
	tmp_0 = np.log2(np.array(C_range))
	plt.plot(tmp_0, tmp, 'b')
	plt.plot(tmp_0, tmp + tmp2, 'g')
	plt.plot(tmp_0, tmp - tmp2, 'g')
	plt.show()
