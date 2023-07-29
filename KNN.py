import numpy as np
import pandas as pd
from sklearn import neighbors , metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/sohamshirke/Documents/Data Science/Codes/car+evaluation/car.data')
data.columns = ['Buying', 'maint', 'doors', 'persons', 'lug_boots', 'safety', 'condition']
print(data.head(10))

X = data[['Buying','maint','safety']].values
X1 = data[['Buying','maint','safety']].values


y = data[['condition']]
print(X)
print(y[0])

print(len(X[0]))
#conersion 
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = Le.fit_transform(X[:,i])
    

print(X)

label_mapping = {'unacc':0 , 'acc':1 , 'good':2 ,'vgood':3 }
y['condition'] = y['condition'].map(label_mapping)
y=np.array(y)
print(y)

#create our model

knn = neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
knn.fit(X_train,y_train)



predict = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test,predict)
print("predict: ",predict)
print("accuracy: ",accuracy )


print('actual value',y[20])
print('predicted value',knn.predict(X)[20])
