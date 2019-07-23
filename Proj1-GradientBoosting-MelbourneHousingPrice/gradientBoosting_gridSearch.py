#Importing modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

# Step-2 Importing dataset
#df = pd.read_csv("E:\VSCODE\Python\ML\Proj1\Melbourne_housing_FULL.csv\Melbourne_housing_FULL.csv")
df = pd.read_csv("C:\\Users\\n0278588\Desktop\\Melbourne_housing_FULL.csv")
#print(df.columns)

# Step-3 Scrubbing
# As part of scrubbing here w eare deleting unwanted columns
del df["Address"]
del df["Method"]
del df["SellerG"]
del df["Date"]
del df["Postcode"]
del df["Lattitude"]
del df["Longtitude"]
del df["Regionname"]
del df["Propertycount"]

##Next scrubbing is do remove the missing values
df.dropna(axis=0, how="any", thresh=None, subset=None, inplace=True)

##Converting Non-Numeric into Nuemeric using Hot-Encoding
features_df = pd.get_dummies(df, columns=["Suburb", "CouncilArea", "Type"])

##Remove the Proce column bcz its going to be our Target(y) and
##currenlty its with our feature(X)
del features_df["Price"]

##Now create X and y arrays from the dataset using .values
X = features_df.values
y = df["Price"].values

#Step-4 Split the Dataset
##We are splitting with standard 70/30 splitting ratio

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True) #Shuffle will not work with scikit-leanr 0.18
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#Step-5 Select ALgorithm and configure hyperparametes
model = ensemble.GradientBoostingRegressor()

#Step-6 Set the configuartion that you wish to test.To menimze the processing time either reduce the no of variables or
## test each hyperparameter separtely
param_grid = {
    'n_estimators' :[300,600],
    'learning_rate': [0.01,0.02],
    'max_depth': [7,9],
    'min_samples_split': [3,4],
    'min_samples_leaf': [5,6],
    'max_features': [0.8,0.9],
    'loss': ['ls','lad','huber']
}

#STep-7 Grid search ,Runs with 4 cpus in parallel
gs_cv = GridSearchCV(model,param_grid,n_jobs=1)

##Run grid search with Trainign data
print("Data modelling in progress..")
gs_cv.fit(X_train,y_train)

#Optimal hyperparameters
print(f"Optimal hyperparameters are {gs_cv.best_params_}")

#Step-8 Evalute the Results
##We are using MAE for evaluting the errors in prediction
mse = mean_absolute_error(y_train,gs_cv.predict(X_train))
print(f"Training Set Mean Absolute Error: {mse}")

##MAE with test Data
mse = mean_absolute_error(y_test,gs_cv.predict(X_test))
print(f"Test Set Mean Absolute Error: {mse}")

print("\nModelling Completed")

if __name__ == "__main__":
    pass
