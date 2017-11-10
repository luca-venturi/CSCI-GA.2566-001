# use Python2

import os
import numpy as np

# write the data from 'spam' into libsvm-type data files

tmp = []
with open('data/spam', 'r') as data_file: # spam_2016 
    tmp.append([line.strip().split(",") for line in data_file])
data = np.array(tmp)[0,:,:].astype(np.float)
print data.shape

nTrain = 3000 # spam_2016 -> 3450     
nTest = 1601 # spam_2016 -> 1151

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
# [-1.,1.] found to be the best choice, according to https://cs.nyu.edu/~mohri/ml16/sol2.pdf

os.system('libsvm/./svm-scale -l -1 -u 1 -s data/train_range data/train > data/train_scaled')
os.system('python checkdata.py data/train_scaled')
os.system('libsvm/./svm-scale -l -1 -u 1 -r data/train_range data/test > data/test_scaled')
os.system('python checkdata.py data/test_scaled')
