import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics

data = np.genfromtxt('../../Dataset/1_Clustering/CSVfiles/D31.csv', delimiter=',',skip_header=1)
label = data[:,-1]
data = data[:,:-1]

kmeans = KMeans(n_clusters=70).fit(data)

y_pred = kmeans.labels_

print("\nRand index : {}".format(metrics.adjusted_rand_score(label,y_pred)))  