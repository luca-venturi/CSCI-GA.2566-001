# Python2

import os
import numpy as np

# write the data from 'spam' into libsvm data files

tmp = []
with open('data/spam', 'r') as data_file:
    tmp.append([line.strip().split(",") for line in data_file])
data = np.array(tmp)[0,:,:].astype(np.float)
print data.shape

nTrain = 3000     
nTest = 1601
nFeat = 57

def data_array_to_file(arr,filename):
	n, m = arr.shape
	os.system('rm -f '+filename)
	with open(filename, 'a') as _file:
		for i in range(n):
			_file.write(str(np.int(arr[i,-1])))
			for j in range(m-1):
				if arr[i,j] != 0:
					_file.write(' '+str(j+1)+':'+str(arr[i,j]))
			_file.write('\n')

print data[:nTrain,:].shape
data_array_to_file(data[:nTrain,:],'data/train')
os.system('python checkdata.py data/train')
print data[-nTest:,:].shape
data_array_to_file(data[-nTest:,:],'data/test')
os.system('python checkdata.py data/test')

# scale the data

os.system('./svm-scale -l 0 -s data/train_range data/train > data/train_scaled')
os.system('python checkdata.py data/train_scaled')
os.system('./svm-scale -l 0 -r data/train_range data/test > data/test_scaled')
os.system('python checkdata.py data/test_scaled')

# train

os.system('./svm-train -t 1 -d 4 -g k -c  -h 0 -v 10 data/train_scaled')
