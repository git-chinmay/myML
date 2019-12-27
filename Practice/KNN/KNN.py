"""
K-Nearest Neighbour
1) 7 Wrong Predictions out of total 100
2) Some strange prediction with input like age = 60,salary=100,output "Will buy SUV"
    Looks like heavily influenceed by Age column.
3) 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\KNN\Social_Network_Ads.csv')

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
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print(f"confusion matrix: {matrix}")

#Random Prediction
agex,salaryx = 60,100
z=[[agex,salaryx]]
z = scale.transform(z)

print(classifier.predict(z))


if classifier.predict(z)[0] == 1:
    print("Will buy the SUV.")
else:
    print("Will not buy the SUV.")

"""
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
plt.title('KNN Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""


    


