import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

fruits =  pd.read_csv("/Users/sohamshirke/Documents/Data Science/Codes/fruit data.csv")
print(fruits.head(10))

X = fruits.iloc[:,3:5].values
y = fruits.iloc[:,0].values
X.shape
y.shape


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn import tree

clf_2 = tree.DecisionTreeClassifier(min_samples_split=2)

clf_2.fit(X_train,y_train)

pred_test= clf_2.predict(X_test)
print(pred_test)
print(y_test)

pred_train = clf_2.predict(X_train)
print(pred_train)

from sklearn.metrics import accuracy_score
acc_test = accuracy_score(y_test, pred_test)
acc_train = accuracy_score(y_train, pred_train)
print(acc_test)
print(acc_train)

# we can see that the model is underfit because of the bias is low 
# where as  the variance is high this tells us that the model is underfit 

fruits
fruits['fruit_name'].value_counts()
# Out[43]: 
# apple       19
# orange      19
# lemon       16
# mandarin     5
# Name: fruit_name, dtype: int64

#here you can add the number of the lemon and mandarin so that the data becomes more vivid than earlier


