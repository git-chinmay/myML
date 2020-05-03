"""
Decision Tree Regression
The DTR predicted salary for 6.5 years: [150000.]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\DecesionTreeRegression\Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor() #default criterion = mse
regressor.fit(X,y)
print(f"The DTR predicted salary for 6.5 years: {regressor.predict(np.array(6.5).reshape(-1,1))}")

#Plot the graph
plt.subplot(1,2,1)
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(np.array(X).reshape(-1,1)),color='blue')
plt.xlabel("Years of Experience")
plt.ylabel('Salary')
plt.title('Experience vs Salary[DTR]-wrong plot')


"""
Above plot will not give correct plotting as the DTR plot should be step wise as each segment of x axis should mainatain
the value for entire segment but here it is increasing linearly.
Lets replot with X_grid
"""
plt.subplot(1,2,2)
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel("Years of Experience")
plt.ylabel('Salary')
plt.title('Experience vs Salary[DTR]')
plt.show()
