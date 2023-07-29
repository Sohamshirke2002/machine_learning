import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


df = datasets.load_iris()
df

X = df.data
y = df.target

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.naive_bayes import GaussianNB

Gb = GaussianNB()
Gb.fit(X_train,y_train)


predict_x = Gb.predict(X_test)
print(predict_x)
print(y_test)

from sklearn.metrics import accuracy_score
accuracy  = accuracy_score(y_test, predict_x)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, predict_x)
