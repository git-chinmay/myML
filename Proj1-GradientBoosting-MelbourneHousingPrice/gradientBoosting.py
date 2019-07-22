# Step-1 Importing libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

# Step-2 Importing dataset

df = pd.read_csv(
    "E:\VSCODE\Python\ML\Proj1\Melbourne_housing_FULL.csv\Melbourne_housing_FULL.csv"
)
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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True)

#Step-5 Select ALgorithm and configure hyperparametes
model = ensemble.GradientBoostingRegressor(
    n_estimators= 150,
    learning_rate= 0.1,
    max_depth= 30,
    min_samples_split= 4,
    min_samples_leaf= 6,
    max_features= 0.6,
    loss= 'huber'

)

##Train the model by using the Fit command
model.fit(X_train,y_train)

#Step-6 Evalute the Results
##We are using MAE for evaluting the errors in prediction
mse = mean_absolute_error(y_train,model.predict(X_train))
print(f"Training Set Mean Absolute Error: {mse}")

##MAE with test Data
mse = mean_absolute_error(y_test,model.predict(X_test))
print(f"Test Set Mean Absolute Error: {mse}")


