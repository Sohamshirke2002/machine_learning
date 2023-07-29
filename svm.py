from sklearn import datasets
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

classes = ['Iris Setosa' , 'Iris Versicolour' , 'Iris Virginica']



from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print(accuracy)
print("actual :" , y_test)
print("prediction :", prediction)
print(model)


for i in range(len(prediction)):
    print(classes[prediction[i]])