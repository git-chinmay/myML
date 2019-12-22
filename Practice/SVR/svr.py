"""
Its from SVM but not SVM.
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\SVR\Position_Salaries.csv')
X= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#We need feature csaling here as SVR dont do it itself like other libraries
from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y.reshape(-1,1))

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1,1))



from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # defaut kernel
regressor.fit(X,y)

y_predict = regressor.predict([[6.5]])
#Converting back to original form
y_predict = scaler.inverse_transform(y_predict)

print(f"The predicted Salary for 6.5 years experience: {y_predict}")

#Plotting the graph
#plt.scatter(X,y,color='red')
#plt.plot(X,regressor.predict(X),color='blue')
#plt.xlabel("Position")
#plt.ylabel('Salary')
#plt.title('Position Vs Salary[SVR]')
#plt.show()

#For smoother curve
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel("Position")
plt.ylabel('Salary')
plt.title('Position Vs Salary[SVR]')
plt.show()

