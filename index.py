import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering



df = pd.read_csv('Mall_Customers.csv')
print(df.head())

#We are not going to use all the columns.
features = df.iloc[:, 2:5]
print(features.head())

#Dendogram
plt.figure(figsize=(100, 50))
dendrogram = sch.dendrogram(sch.linkage(features, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

"""If I want 3 clusters I will draw a horizontal line from 300 in the Y-axis. If I want 2 clusters 
I will draw a horizontal line from 400 in the Y-axis"""

plt.figure(figsize=(100, 50))
dendrogram = sch.dendrogram(sch.linkage(features, method='ward'))
plt.axhline(y=300, color='r')
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

"""there are two variants of it, one named Agglomerative Hierarchical Clustering and the other named Divisive Hierarchical Clustering.
If we move from the bottom of the dendrogram from many clusters to one cluster it is known as Agglomerative Hierarchical Clustering. 
Inversely,If you move from the top of the dendrogram from one cluster to many clusters we call it Divisive Hierarchical Clustering."""

#CLUSTER

# 3 clusters
cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
x = cluster.fit_predict(features)
print(x)

plt.figure(figsize=(100, 50))
plt.scatter(features['Annual Income (k$)'], features['Spending Score (1-100)'], c=cluster.labels_)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# 4 clusters.
cluster = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
cluster.fit_predict(features)

plt.figure(figsize=(100, 50))
plt.scatter(features['Annual Income (k$)'], features['Spending Score (1-100)'], c=cluster.labels_)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()