"""
K-Means clustering is Unupervised Machine Learning Mechanism
Here we have dataset from a shoping mall which holds data 
    -Customer AGe,Salary and Shopping score
We have to catagorise them based on theor Shopping and credit score
W dont have any idea what will be the grouping look like hence we are using unsupervised learning.
We dont have any Lebeled target for comparision.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\n0278588\GITHUB-Local\myML\Practice\13-KMeansClustering\Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values #Age and Annual Income

#No need to data spliting
#Lets perform elbow method to get K value
from sklearn.cluster import KMeans
"""
wcss = within cluster sum square
init = random intialisation (use kmeans++ to prevent trap)
"""
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#Visualising the elbow plot
plt.plot(range(1,11),wcss)
plt.xlabel("K values")
plt.ylabel("wcss value")
plt.title("Elbow Plot")
plt.show()


#Now running model with K=5
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualising
"""My X having 2 columns,scatter needs X and y,
so X[y_kmeans == 0,0] pointing to values of '0'th column of X for which y_kmeans = 0
and X[y_kmeans == 0, 1] poininting to values of 1st column of X where y_kmenas = 0 
We have draw similarly for 5 clusters as K = 5 we have choosen."""

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Low Spending Customer')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Mid Spending Customer')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target Customer')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless Customer')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible Customer')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

