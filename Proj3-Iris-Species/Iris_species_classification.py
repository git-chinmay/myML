"""
Performing the Linear classification on the Iris flower dataset
"""

from sklearn import datasets,preprocessing
#from sklearn.cross_validation import train_test_split #cros validation is no more used in sklearn librray.Use model seelction instaed
from sklearn.model_selection import train_test_split



iris = datasets.load_iris()
X_iris,y_iris = iris.data,iris.target

#print(X_iris[0:5, :2]) #We are using only 2 features out total 4 features
X,y = X_iris[:, :2],y_iris

#Split the data into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 33)
##print(X_train.shape,y_train.shape)
##(112, 2) (112,)

#Standardise the feature

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Now lest go ahead and plot the training set to see how instances distributed 
import matplotlib.pyplot as plt
colors = ['red','greenyellow','blue']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train==i]
    ys = X_train[:, 1][y_train==i]
    plt.scatter(xs,ys,c= colors[i])

plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

##Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train,y_train)
print(f"The coeeficient {clf.coef_}")
print(f"The intercept {clf.intercept_}")

