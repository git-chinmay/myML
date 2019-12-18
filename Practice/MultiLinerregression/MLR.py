"""
Link to bakward eliminating concept
https://www.geeksforgeeks.org/ml-multiple-linear-regression-backward-elimination-technique/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

dataset = pd.read_csv(r'C:\Users\n0278588\GITHUB-Local\myML\Practice\MultiLinerregression\50_Startups.csv')

#Data preprocessing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#categorical data hot encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable Trap
X = X[:,1:]

#Split the data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)


#Trainign the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict the y using test input data 
y_predict = regressor.predict(X_test)

print(f"y-test data: {y_test}")
print(f"Model predicted data: {y_predict}")

"""
y-test data: [103282.38 144259.4  146121.95  77798.83 191050.39 105008.31  81229.06
  97483.56 110352.25 166187.94]
Model predicted data: [103015.20159795 132582.27760816 132447.73845175  71976.09851258
 178537.48221056 116161.24230166  67851.69209676  98791.73374687
 113969.43533013 167921.06569552]
"""


