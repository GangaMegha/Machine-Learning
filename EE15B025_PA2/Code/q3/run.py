import csv
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Colormap
cmap = colors.LinearSegmentedColormap(
    'flower',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 1, 1), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

alpha = 0.05 # reg-param for RDA

def plot(indx, title):
	
	plt.figure(indx)
	plt.title(title)
	plt.xlabel("Pixel Length")
	plt.ylabel("Pixel Width")
	
	if(title=="LDA"):
		lda = LinearDiscriminantAnalysis(solver = "svd")
		lda.fit(data, label_encode)
	elif(title=="QDA"):
		qda = QuadraticDiscriminantAnalysis()
		qda.fit(data, label_encode)
	elif(title=="RDA"):
		qda = QuadraticDiscriminantAnalysis(reg_param=alpha)
		qda.fit(data, label_encode)
	
	# Draw decision boundary
	x_m, x_M, y_m, y_M = data[:,0].min()-1, data[:,0].max()+1, data[:,1].min()-1, data[:,1].max()+1
	xx, yy = np.meshgrid(np.arange(x_m, x_M, 0.01), np.arange(y_m, y_M, 0.01))
	if(title=="LDA"):
		Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = np.reshape(Z, xx.shape)
		plt.pcolormesh(xx, yy, Z, cmap="flower")

	elif(title=="QDA" or title=="RDA"):
		Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = np.reshape(Z, xx.shape)
		plt.pcolormesh(xx, yy, Z, cmap="flower")
	plt.contour(xx, yy, Z, alpha=0.4)

	# Plot the points
	plt.scatter(data[np.where(labels=="Iris-setosa"),0], data[np.where(labels=="Iris-setosa"),1], marker="+", c="b", cmap=plt.cm.spectral)
	plt.scatter(data[np.where(labels=="Iris-versicolor"),0], data[np.where(labels=="Iris-versicolor"),1], marker="x", c="g", cmap=plt.cm.spectral)
	plt.scatter(data[np.where(labels=="Iris-virginica"),0], data[np.where(labels=="Iris-virginica"),1], marker="*", c="r", cmap=plt.cm.spectral)

	# Plot mean of each class
	if(title=="LDA"):
		mean = np.array(lda.means_)
		# print(lda.explained_variance_ratio_)
	elif(title=="QDA" or title=="RDA"):
		mean = np.array(qda.means_)

	plt.scatter(mean[:,0], mean[:,1], marker="o", c=["b", "g", "r"], cmap=plt.cm.spectral, edgecolors="k")
	for k, label, c in zip(mean, np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"]), np.array(["b", "g", "r"])):
		plt.text(k[0]+0.2, k[1]-0.59, label, color=c)


data = list(csv.reader(open("../../Dataset/iris.data.csv", "rb"), delimiter = ","))

data = data[:-1]
np.random.shuffle(data)

labels = np.array([data[row][4] for row in range(len(data))])
data = np.array([(data[row][2], data[row][3]) for row in range(len(data))]).astype("float")

# encode the labels as integers 
label_encode = np.zeros(labels.shape).astype("int")
label_encode[np.where(labels=="Iris-setosa")] = 0
label_encode[np.where(labels=="Iris-versicolor")] = 1
label_encode[np.where(labels=="Iris-virginica")] = 2

plot(1,"LDA")
plot(2,"QDA")
plot(3,"RDA")

plt.show()