import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\SimpleLinearregression\Salary_Data.csv')
#print(dataset.head(5))

##DATA PREPROCESSING SECTION
#No empty rows
#Lets get the X and y from dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#No need to do feature scaling
#Now lets split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


##Train the model
#Fiting the linear model with trainging data
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_predict = regressor.predict(X_test)
print(f"Predicted value {y_predict}")
print(f"Actual value {y_test}")

##Plotting the Graph
#Plot Trainigset of data and experirnce
pyplot.scatter(X_train,y_train,color="red")
pyplot.plot(X_train,regressor.predict(X_train),color="blue") #The regressor line for train data
pyplot.xlabel("Years of experience")
pyplot.ylabel("Salary")
pyplot.title("Salary vs Experience[Training Set]")
pyplot.show()

#Plot Testset of data and experirnce
pyplot.scatter(X_test,y_test,color="red")
pyplot.plot(X_train,regressor.predict(X_train),color="blue") #We are using same Xtrain because we wanr to use same uniq model equation for comparing the with new test data
pyplot.xlabel("Years of experience")
pyplot.ylabel("Salary")
pyplot.title("Salary vs Experience[Test Set]")
pyplot.show()
