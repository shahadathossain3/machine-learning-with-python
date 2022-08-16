import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(X)

print(y)

y=y.reshape(len(y),1)

print(y)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

print(X)
print(y)

from sklearn.svm import SVR
regressor=SVR(kernel= 'rbf')
regressor.fit(X, y)
