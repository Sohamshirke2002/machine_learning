from sklearn import datasets
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)


data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target

X.shape

lr = linear_model.LinearRegression()

X_train ,X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
model  = lr.fit(X_train, y_train)

predicttion = model.predict(X_test)
print(predicttion)
print("r2 value ", lr.score(X, y))
print("coeff" , lr.coef_)
print("iterscept ",lr.intercept_)


