import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

companies = pd.read_csv('/Users/sohamshirke/Downloads/1000_Companies.csv');
companies.head(10)

X = companies.iloc[:,:-1].values
Y = companies.iloc[:,4].values

print(X) #here there are x1 , x2 , x3
print(Y) #it has the profit 

sns.heatmap(companies.corr()) #corr is for the coordinates

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])


from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)


#Dummy data Trap avoid it 
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

# Calculating the Coefficients
print(regressor.coef_)

# Calculating the Intercept
print(regressor.intercept_)

# Calculating the R squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
