# Python3

from sklearn.tree import DecisionTreeClassifier

# load data

with open ('data/data_python', 'rb') as _file:
    [xTrain,yTrain,xTest,yTest] = pickle.load(_file)

# model

cl = DecisionTreeClassifier(max_depth=1)
