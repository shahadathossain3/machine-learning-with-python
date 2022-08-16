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
y=np.ravel(y)
regressor.fit(X, y)

preduct_new_result=sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
print(preduct_new_result)

plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
