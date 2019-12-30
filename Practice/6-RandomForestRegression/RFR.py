"""
Random Forest Regression Model.(Combination of bunch of DTRs)
Observations:
    The RFR predicted salary for 6.5 years: [147000.] with 10 trees 
    The RFR predicted salary for 6.5 years: [158100.] with 100 trees
    The RFR predicted salary for 6.5 years: [160333.33333333] with 300 trees (Bull's Eye)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\DecesionTreeRegression\Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import  RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300) #default value os 10
regressor.fit(X,y)
print(f"The RFR predicted salary for 6.5 years: {regressor.predict(np.array(6.5).reshape(-1,1))}")


X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel("Years of Experience")
plt.ylabel('Salary')
plt.title('Experience vs Salary[RFR]')
plt.show()
