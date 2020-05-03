"""
Decision Tree Classification.(DT can be used in both Regression and classification problem)
In DT actually we dont need feature scaling but still we are going to use it as we are visualising the 
output with some resolution setting and without featur scalingit will may no work or 
take too much time to process.

Observations:
--9 wrong predictions
--With agex,salaryx = 60,1 : will buy the SUV
--With Criterion = 'gini' we have 10 wrong preictions

Be careful,from the training dataset visulaisation it looks lime model doing overfitting
Its trying to capture all the DP in a region but while running test data.same region are empty.
Over all the model looks good with less wrong predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\8-KNN\Social_Network_Ads.csv')

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.25,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

#Model training
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print(f"confusion matrix: {matrix}")


#Random Prediction
agex,salaryx = 60,1
z=[[agex,salaryx]]
z = scale.transform(z)

print(classifier.predict(z))


if classifier.predict(z)[0] == 1:
    print("Will buy the SUV.")
else:
    print("Will not buy the SUV.")


#Visualisation
from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decesion Tree Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


    


