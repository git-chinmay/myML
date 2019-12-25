"""
Logistic Regression
Observations :
1)Taking Userid and Age and saalry as Input and without Feature scaling
    No of wrong predictions: 19 out of total 100 observations.

2)Taking Userid and Age and saalry as Input and with Feature scaling
    No of wrong predictions: 14 out of total 100 observations.

3)All inputs with One hot encoding
    No of wrong predictions: 11 out of total 100 observations.


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\n0278588\GITHUB-Local\myML\Practice\LogisticRegression\Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values #May need hot encoding for Gender column
#X = dataset.iloc[:,[0,2,3]].values # By this we can select specifc columns
y = dataset.iloc[:,4].values

#Hot encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
X[:,1] = encoder.fit_transform(X[:,1])
hotencoder = OneHotEncoder()
X= hotencoder.fit_transform(X).toarray()



#Splitting the data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(f"Predicted value: {y_pred}")

#for i in y_pred:
#    print(i)

print(f"Actual value: {y_test}")
#for j in y_test:
#    print(j)

#Checking no of wrong prediction
length_of_array = len(y_pred)
match = 0
for i in range(length_of_array):
    if y_pred[i] != y_test[i]:
        match +=1

print(f"\nNo of wrong predictions: {match} out of total {length_of_array} observations.")




