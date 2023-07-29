from sklearn import datasets 
import numpy as np

iris = datasets.load_iris()

#split in features and labels
X = iris.data
y = iris.target

#4 features and one target
print(X)
print(y)

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)

