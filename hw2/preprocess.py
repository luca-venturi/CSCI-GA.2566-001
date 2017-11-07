# Python2

import os
import numpy as np
from svmutil import *

# write the data from 'spam.data' into libsvm data files

tmp = []
with open('spam.data', 'r') as data_file:
    tmp.append([line.strip().split(",") for line in data_file])
data = np.array(tmp)[0,:,::-1]

nTrain = 3000     
nTest = 1601
nFeat = 57

def data_array_to_file(arr,filename):
	n, m = arr.shape
	os.system('rm -f '+filename)
	with open(filename, 'a') as _file:
		for i in range(n):
			_file.write(str(arr[i,0]))
			for j in range(1,m):
				_file.write(' '+str(j)+':'+str(arr[i,j]))
			_file.write('\n')
				
data_array_to_file(data[:nTrain,:],'data.train')
data_array_to_file(data[-nTest:,:],'data.test')

# scale the data

os.system('./svm-scale -s data.train.range data.train > data.train.scale')
os.system('./svm-scale -r data.train.range data.test > data.test.scale')


# get the scaled data as array

def data_file_to_array(filename):
	arr = []
	data_len = []
	ll = []
	i = 0
	with open(filename, 'r') as _file:
		for line in _file:
			tmp = line.strip().split()
			row = [np.int(tmp[0])]
			for val in tmp[1:]:
				row.append(np.float(val.split(':')[1]))
			arr.append(row)
			data_len.append(len(row))
	return np.asarray(arr)
	
dtrs = data_file_to_array('data.train.scale')

# compute a random split of the training data

nBatch = 10
nTrainBatch = np.int(nTrain / nBatch)
tmp = np.arange(nTrain)
np.random.shuffle(tmp)
ind = [list(tmp[i*nTrainBatch:(i+1)*nTrainBatch]) for i in range(nBatch)]
#drtsBatch = np.zeros((,nBatch))


