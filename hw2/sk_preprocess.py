import numpy as np
# use Python3

import pickle
import sklearn.preprocessing as pr

# write data from 'data/spam' into np.array

tmp = []
with open('data/spam_2016', 'r') as data_file:
    tmp.append([line.strip().split(",") for line in data_file])
data = np.array(tmp)[0,:,:].astype(np.float)

# get Train and Test set

nTrain = 3450#3000     
nTest = 1151#1601

xTrain = data[:nTrain,:-1]
yTrain = data[:nTrain,-1].astype(np.int)
xTest = data[-nTest:,:-1]
yTest = data[-nTest:,-1].astype(np.int)

# scale data

maxRange = 1.
scaler = pr.MinMaxScaler(feature_range=(-maxRange, maxRange))
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# save data to file 

_list = [xTrain,yTrain,xTest,yTest]

with open('data/data_python', 'wb') as _file:
    pickle.dump(_list, _file)
