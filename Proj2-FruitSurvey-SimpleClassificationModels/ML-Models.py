"""
Testing various Supervised Learning ML models using a simple dataset
Practice link :- 
    https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2

"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  #To avoid future warnings in output

fruit = pd.read_table(r"C:\Users\n0278588\GITHUB-Local\myML\Proj2-FruitSurvey-SimpleClassificationModels\InputDataSet.txt")
#print(fruit.head())
#print(fruit.shape)
#print(fruit['fruit_name'].unique())
#print(fruit.groupby('fruit_name').size())



###The input data visulaisation 
import seaborn as sns 
#sns.countplot(fruit['fruit_name'],label = "Count")
#plt.show()



###Visualizing the input variable distribution

#fruit.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
#                                        title='Box Plot for each input variable')
#plt.savefig('fruits_box')
#plt.show()



#visualising same in Histogram 
import pylab as pl
#fruit.drop('fruit_label',axis =1).hist(bins=30,figsize=(9,9))
#pl.suptitle("Histogram for each numeric input variable")
#plt.savefig('fruits_hist')
#plt.show()





#Creating scatter matrix for each input variable
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
#
#feature_names = ['mass','width','height','color_score']
#X = fruit[feature_names]
#y = fruit['fruit_label']
#
#cmap = cm.get_cmap('gnuplot')
#scatter = pd.scatter_matrix(X, c=y, marker = 'o', s=40, hist_kwds=('bins':15),figsize=(9,9),cmap=cmap)
#plt.suptitle('Scatter matrix for each input varible')
#plt.savefig('fruits_scatter_matrix')


###Creating Training and Test sets and Apply Scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

feature_names = ['mass','width','height','color_score']
X = fruit[feature_names]
y = fruit['fruit_label']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


###Building Models###

##Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
print(f'\nAccuracy of Logistit regression on Train data {logreg.score(X_train,y_train):.2f}')
print(f'Accuracy of Logistit regression on Test data {logreg.score(X_test,y_test):.2f}')


##Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

print(f'\nAccuracy of Decision Tree Classifier on Train data {clf.score(X_train,y_train):.2f}')
print(f'Accuracy of Decision Tree Classifier on Test data {clf.score(X_test,y_test):.2f}')

##K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(f'\nAccuracy of K-NN on Train data {knn.score(X_train,y_train):.2f}')
print(f'Accuracy of K-NN on Test data {knn.score(X_test,y_test):.2f}')

##Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
print(f'\nAccuracy of Linear Discriminat Analysis on Train data {lda.score(X_train,y_train):.2f}')
print(f'Accuracy of Linear Discriminat Analysis on Test data {lda.score(X_test,y_test):.2f}')

##Gaussian naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
print(f'\nAccuracy of GNB classifier on Train data {gnb.score(X_train,y_train):.2f}')
print(f'Accuracy of GNB classifier on Test data {gnb.score(X_test,y_test):.2f}')

##Support Vector machine
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
print(f'\nAccuracy of SVM on Train data {svm.score(X_train,y_train):.2f}')
print(f'Accuracy of SVM classifier on Test data {svm.score(X_test,y_test):.2f}')

##Prediction using K-NN model
from sklearn.metrics import classification_report,confusion_matrix
pred = knn.predict(X_test)
#print(pred)
print("")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
