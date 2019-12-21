"""
Polynomial Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\n0278588\GITHUB-Local\myML\Practice\Polynomialregression\Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#NOTE: If we use this format then X_grid reshaping cant be done because of type error issue min(X)
#X = dataset[["Level"]] #converting into a 2D array
#y = dataset["Salary"]


"""
No missing values present
No need to do feature scalling as python libary will do it internally if needed
No need to split the data as we have only 10 observations.
"""
#Lets first create a simple liner regression
from sklearn.linear_model import LinearRegression
smp_regressor = LinearRegression()
smp_regressor.fit(X,y)
y_predict=smp_regressor.predict(X)

#Plotting the data 
plt.subplot(1,3,1)
plt.scatter(X,y,color='red')
plt.plot(X,y_predict,color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')


#Doing ploynomial regression
#First convert the input X to polynomial order(degree)
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=5) #after 5 you will dont see any change in curve fitting ,hence 5th is the best fit
X_poly = pf.fit_transform(X) #It will automatically add a column of 1's
pf.fit(X_poly,y)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly,y)


#plotting the Polynomial data 
plt.subplot(1,3,2)
plt.scatter(X,y,color='red')
plt.plot(X,poly_regressor.predict(X_poly),color="blue")
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level(Poly)')
plt.ylabel('Salary')


#Reshaping the grid
X_grid = np.arange(min(X),max(X),0.1) # Create a list of 1 to 9.9
X_grid = X_grid.reshape(len(X_grid),1) #Reshaping to length of grid rows and 1 column
plt.subplot(1,3,3)
plt.scatter(X,y,color='red')
plt.plot(X_grid,poly_regressor.predict(pf.fit_transform(X_grid)),color="blue")
plt.title('Truth or Bluff (Polynomial Regression) with High resolution')
plt.xlabel('Position level(Poly)')
plt.ylabel('Salary')



#Predicting a new result with Liner regression
#Both print same.Both are converting 1 sclar value 6.5 to a 2D array.
print("Predicting a new result with Liner regression")
print(smp_regressor.predict(np.array(6.5).reshape(1,-1)))
print(smp_regressor.predict([[6.5]]))

#predicting a new result with Polynomial Regression
print("\nPredicting a new result with Polynomial Regression")
print(poly_regressor.predict(pf.fit_transform(np.array(6.5).reshape(1,-1))))
print(poly_regressor.predict(pf.fit_transform([[6.5]])))

plt.show()
