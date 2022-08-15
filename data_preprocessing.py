import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('Data.csv')
# print(dataset)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:, 1:3])

X[:,1:3]=imputer.transform(X[:,1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encode', OneHotEncoder(),[0])], remainder='passthrough')

X=np.array(ct.fit_transform(X))

print(X)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

print(y)

