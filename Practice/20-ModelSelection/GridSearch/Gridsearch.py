# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:15:01 2020
@author: Chinmay
Grid Search:- To choose the optimum value for hyperparameters for best performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\20-ModelSelection\K-Fold_CV\Social_Network_Ads.csv')

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
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print(f"confusion matrix: {matrix}")

#K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()

#Applying Grid Search to find best model and its best Parameter
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000],'kernel':['linear']},
               {'C':[0.2,0.4,0.8,1],'kernel':['rbf'],'gamma':[0.5,0.7,0.9,1.0]}
        ]

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)#n_jobs for large datasets -1 to use all power of ur system

grid_search=grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_

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
plt.title('SVM Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


    





