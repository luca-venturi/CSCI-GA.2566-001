# use Python3

import numpy as np
import pickle
import sklearn.preprocessing as pr

# write data from 'data/spam' into np.array

tmp = []
with open('data/spam', 'r') as data_file: # spam_2016
    tmp.append([line.strip().split(",") for line in data_file])
data = np.array(tmp)[0,:,:].astype(np.float)

# get Train and Test set

nTrain = 3000 # spam_2016 -> 3450     
nTest = 1601 # spam_2016 -> 1151

xTrain = data[:nTrain,:-1]
yTrain = data[:nTrain,-1].astype(np.int)
xTest = data[-nTest:,:-1]
yTest = data[-nTest:,-1].astype(np.int)

# scale data

maxRange = 1.0 # 0.1 for my_kernel to work in reasonable time
scaler = pr.MinMaxScaler(feature_range=(-maxRange, maxRange)) # found to be the best choice, according to https://cs.nyu.edu/~mohri/ml16/sol2.pdf
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# save data to file 

_list = [xTrain,yTrain,xTest,yTest]

with open('data/data_python', 'wb') as _file:
    pickle.dump(_list, _file)
