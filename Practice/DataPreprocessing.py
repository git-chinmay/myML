import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

dataset = pd.read_csv(r"E:\VSCODE\GIT_Hub\myML\Practice\Data.csv")

#create matrix of feature and y vectors
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#To remove the null values
"""
from sklearn.preprocessing  import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
#print(X)"""

#If you receive the depreciation warning for Imputer
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
missingvalues = missingvalues.fit(X[:,1:3])
X[:,1:3] = missingvalues.transform(X[:,1:3])


#To handel the categorical data
#step 1 Label encoder
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
#print(X)

#step-2 lets use hot encoder to avoid misunderstanding in labelencoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder_x = OneHotEncoder(categorical_features=[0])
X= onehotencoder_x.fit_transform(X).toarray()
#print(X)

#As purchase column is dependant only labelencoder will work
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)
print(y)
