import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.utils import shuffle

def _preprocess(data_set, random_state=1024, frac=0.5):
    with open('data/uci/' + data_set + '.data', 'r') as _file:
        data_array = []
        for line in _file:
            tmp = line.rstrip().split(',')
            data_array.append(tmp)
        data_array = np.array(data_array)
        features = data_array[:, :-1].astype(np.float)
        labels = data_array[:, -1]
        classes = list(set(labels))
        if data_set != 'kin8nm':
            for i in range(labels.shape[0]):
                if labels[i] == classes[0]:
                    labels[i] = 1
                else:
                    labels[i] = -1
        labels = labels.astype(np.float)
        features, labels = shuffle(features, labels, random_state=random_state)
        frac = int( features.shape[0] * frac )
        xTrain = features[:frac, :]
        xTest = features[frac:, :]
        yTrain = labels[:frac]
        yTest = labels[frac:]
        scaler = preprocessing.MinMaxScaler(feature_range=(0., 1.))
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)
        mTrain = np.mean(xTrain,axis=0)
        xTrain -= mTrain
        xTest -= mTrain
        _list = [xTrain, yTrain, xTest, yTest]
        with open('data_python/' + data_set, 'wb') as _file:
            pickle.dump(_list, _file)
            
if __name__ == '__main__':
    data_sets = ['sonar', 'ionosphere']
    for dataset in data_sets:
        _preprocess(dataset)
