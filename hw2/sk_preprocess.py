import numpy as np
# use Python3

import pickle
from sklearn.preprocessing import StandardScaler

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

# scale data

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# save data to file 

_list = [xTrain,yTrain,xTest,yTest]

with open('data/data_python', 'wb') as _file:
    pickle.dump(_list, _file)
