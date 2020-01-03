"""
Agglomerative is a type of Hierichal clustering mechanism.
It uses Dendogram to find the optimum no of clusters like K-Means uses WCSS

For Unseen data prediction 

print(classifier.fit_predict([[1,10]]))
Error: Found array with 1 sample(s) (shape=(1, 2)) while a minimum of 2 is required by AgglomerativeClustering.

print(classifier.fit_predict([[1,10],[2,20]]))
Error: Cannot extract more clusters than samples: 5 clusters where given for a tree with 2 leaves.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\n0278588\GITHUB-Local\myML\Practice\13-KMeansClustering\Mall_Customers.csv')

X =dataset.iloc[:,[3,4]].values
#Unsupervised learning hence there will be no y
#No feature scaling needed


#Dendogram
import scipy.cluster.hierarchy as sch
plt.subplot(1,2,1)
y = sch.linkage(X,'ward')
dn = sch.dendrogram(y)
plt.title("Dendogram Plot")
plt.ylabel("Equlidian Distance")
plt.xlabel("DataPoints or Clusters")


#Fitting the Model
from sklearn.cluster import AgglomerativeClustering
classifier = AgglomerativeClustering(n_clusters = 5,affinity='euclidean',linkage='ward')
y_hclustering= classifier.fit_predict(X)

#print(classifier.fit_predict([[1,10],[2,20]]))



#Visualising
plt.subplot(1,2,2)
plt.scatter(X[y_hclustering == 0, 0], X[y_hclustering == 0, 1], s = 100, c = 'red', label = 'Careful Customers')
plt.scatter(X[y_hclustering == 1, 0], X[y_hclustering == 1, 1], s = 100, c = 'blue', label = 'Standard Customers')
plt.scatter(X[y_hclustering == 2, 0], X[y_hclustering == 2, 1], s = 100, c = 'green', label = 'Target Customer')
plt.scatter(X[y_hclustering == 3, 0], X[y_hclustering == 3, 1], s = 100, c = 'cyan', label = 'Careless Customer')
plt.scatter(X[y_hclustering == 4, 0], X[y_hclustering == 4, 1], s = 100, c = 'magenta', label = 'Sensible Customer')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
