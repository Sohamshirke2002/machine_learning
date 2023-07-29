import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("/Users/sohamshirke/Documents/Data Science/Codes/diabetes.csv")
df

df.columns
X = df.iloc[:,0:8].values
y=df.iloc[:,-1].values

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split
X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size=0.2)

logr = LogisticRegression()
model = logr.fit(X_train, y_train)

prediction = model.predict(X_test)
print(prediction)

print(y_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, prediction)
