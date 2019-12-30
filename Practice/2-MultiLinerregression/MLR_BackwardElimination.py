import numpy as np
import pandas as pd

#STEP-1 DATA PreProcessing
#dataset = pd.read_csv(r"C:\Users\n0278588\GITHUB-Local\myML\Practice\MultiLinerregression\50_Startups.csv")
dataset = pd.read_csv(r"E:\VSCODE\GIT_Hub\myML\Practice\MultiLinerregression\50_Startups.csv")

#defining the input feature matrix and output vector
#print(dataset.columns)
X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State']] #2D Array
y = dataset[['Profit']]


#Doing hot encoding
X = pd.get_dummies(X)

#Splitting the data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#STEP-2 Train the model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict the data 
y_predict = regressor.predict(X_test)

#print(f"Test dataset: {y_test}")
#print(f"Predicted dataset: {y_predict}")

#STEP-3 Using backward elimination Technique to understand which feature is the important factor
#import statsmodels.formula.api as sm

#Adding an extra column of 1s 
#50 rows,1 column 
#arr should be arr = X but it will add new column 1 to right side of X but we need it to be added as 1st column
#Hence we reversed it and value = X now.It will add our X to column of 1s
X = np.append(arr = np.ones((50,1)).astype(int),values=X, axis=1)

#Choose a Significance level usually 0.05
#If P>0.05 for the highest values parameter,remove that value

##Faced errors at this stage
#Error-1 : from scipy.misc import factorial ImportError: cannot import name 'factorial'
#Solution: Reduced one level of scipy version e.g version: scipy==1.2.0 (Existing was 1.3.0)

#Error-2 : from scipy.stats.stats import ss ImportError: cannot import name 'ss'
#Solution: Upgrade the stasmodel "pip install statsmodels==0.10.0rc2 --pre"
#Note:- Dont downgrade the scipy

"""
x_opt = X[:,[0,1,2,3,4,5]]
ols = sm.ols(endog = y, exog= x_opt).fit()
ols.summary()"""

##Error-ols = sm.ols(endog = y, exog= x_opt).fit() TypeError: from_formula() missing 2 required positional arguments: 'formula' and 'data'
#Solution :-  The library they are using is not where the OLS function resides any longer.

import statsmodels.regression.linear_model as lm

#Creating an optimal matrices of features

x_opt = X[:,[0,1,2,3,4,5]]

#OLS :- Ordinary Least Square
#regressor_ols = lm.OLS(endog = y, exog= x_opt).fit()
#print(regressor_ols.summary())

#Remove the 4th(index 4) from  x_opt = X[:,[0,1,2,3,4,5]] column as x4 = 0.990 in previous run
x_opt = X[:,[0,1,2,3,5]]
#regressor_ols = lm.OLS(endog = y, exog= x_opt).fit()
#print(regressor_ols.summary())

#Remove the 5th column(Index 4) from x_opt = X[:,[0,1,2,3,5]] as x4 = 0.940
x_opt = X[:,[0,1,2,3]]
#regressor_ols = lm.OLS(endog = y, exog= x_opt).fit()
#print(regressor_ols.summary())

#Remove the 3rd Column(Index 2) from x_opt = X[:,[0,1,2,3]] as x2 = 0.6
x_opt = X[:,[0,1,3]]
#regressor_ols = lm.OLS(endog = y, exog= x_opt).fit()
#print(regressor_ols.summary())

#Remove the 3rd column(index 2) from x_opt = X[:,[0,1,3]] as x2 = 0.06
x_opt = X[:,[0,1]]
regressor_ols = lm.OLS(endog = y, exog= x_opt).fit()
print(regressor_ols.summary())

##Verdict :- Now we have only column x0 which is column of 1s and x1 which is Original column R & D 
#So it looks like the most impact ful column is R & D in preditcing the y









