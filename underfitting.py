import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df = pd.read_csv("/Users/sohamshirke/Documents/Data Science/Codes/Position_Salaries.csv")
df

X = df.iloc[:,1].values
y = df.iloc[:,2].values
X=X.reshape(-1,1)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X,y)

from sklearn.model_selection import train_test_split
X_train , X_test, y_train , y_test = train_test_split(X,y,)

plt.scatter(X, y, color='red')
plt.plot(X,lr.predict(X),color='blue')
plt.title('Linear Regerssion')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# here you can see that the model used here is simple (that is linear regerssion) where as the 
# where as the data is Non linear you can see it in the grpah itself
# so model is underfiit so it cant learn from the data

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
print("actual values of X :"  )
print(X)
print("After Polynomial regression of degree : 3")
print(X_poly)

lr2 = LinearRegression()
lr2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X,lr2.predict(X_poly),color='blue')
plt.title('Linear Regerssion')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()