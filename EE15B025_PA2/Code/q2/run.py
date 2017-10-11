import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from numpy import newaxis

from sklearn import decomposition

from main import train_data, train_labels, test_data, test_labels, X_train, X_test
from main import regression_object, X_train_intercept, X_test_intercept, lda

cmap = colors.LinearSegmentedColormap(
    'mesh',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 1, 1), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

#3D plot of train dataset
fig_train = plt.figure(1)
plt.clf()
ax_train = Axes3D(fig_train, rect=[0, 0, .95, 1], elev=4, azim=116)

plt.cla()
plt.title("3D plot of Train dataset")
for name, label in [('Class1', 1), ('Class2', 2)]:
    ax_train.text3D(train_data[train_labels[:,0] == label, 0].mean(),
        train_data[train_labels[:,0] == label, 1].mean(),
        train_data[train_labels[:,0] == label, 2].mean(), name,
        horizontalalignment='center',
        bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

ax_train.w_xaxis.set_ticklabels([])
ax_train.w_yaxis.set_ticklabels([])
ax_train.w_zaxis.set_ticklabels([])

train_color = np.choose(train_labels, ["b", "r", "g"])
ax_train.scatter(train_data[:,0], train_data[:,1], train_data[:,2], c=train_color, cmap=plt.cm.spectral, edgecolor='k')


#3D plot of train dataset with LDA decision boundary
fig_train = plt.figure(2)
plt.clf()
ax_train = Axes3D(fig_train, rect=[0, 0, .95, 1], elev=2, azim=118)

plt.cla()
plt.title("3D Plot of Train Dataset : LDA Decision Boundary")

for name, label in [('Class1', 1), ('Class2', 2)]:
	ax_train.text3D(train_data[train_labels[:,0] == label, 0].mean(),
    	train_data[train_labels[:,0] == label, 1].mean(),
    	train_data[train_labels[:,0] == label, 2].mean(), name,
    	horizontalalignment='center',
    	bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Plot class 2 
ax_train.scatter(train_data[np.where(train_labels==2),0], train_data[np.where(train_labels==2),1], train_data[np.where(train_labels==2),2], c="g", cmap=plt.cm.spectral, edgecolor='k')

# Plot the decision boundary
a = lda.coef_[0][0]
b = lda.coef_[0][1]
c = lda.coef_[0][2]
d = lda.intercept_

x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
zz = (-a*xx - b*yy - d)*1./c
ax_train.plot_surface(xx,yy,zz, color='yellow')

# Plot class 1 
ax_train.scatter(train_data[np.where(train_labels==1),0], train_data[np.where(train_labels==1),1], train_data[np.where(train_labels==1),2], c="r", cmap=plt.cm.spectral, edgecolor='k')

ax_train.w_xaxis.set_ticklabels([])
ax_train.w_yaxis.set_ticklabels([])
ax_train.w_zaxis.set_ticklabels([])



# 3D plot of test dataset
fig_test = plt.figure(3)
plt.clf()
ax_test = Axes3D(fig_test, rect=[0, 0, .95, 1], elev=4, azim=116)

plt.cla()
plt.title("3D plot of Test Dataset")
for name, label in [('Class1', 1), ('Class2', 2)]:
    ax_test.text3D(test_data[test_labels[:,0] == label, 0].mean(),
        test_data[test_labels[:,0] == label, 1].mean(),
        test_data[test_labels[:,0] == label, 2].mean(), name,
        horizontalalignment='center',
        bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

test_color = np.choose(test_labels, ["b", "r", "g"])
ax_test.scatter(test_data[:,0], test_data[:,1], test_data[:,2], c=test_color, cmap=plt.cm.spectral, edgecolor='k')

ax_test.w_xaxis.set_ticklabels([])
ax_test.w_yaxis.set_ticklabels([])
ax_test.w_zaxis.set_ticklabels([])



# 3D plot of test dataset with LDA decision boundary
fig_test = plt.figure(4)
plt.clf()
ax_test = Axes3D(fig_test, rect=[0, 0, .95, 1], elev=2, azim=118)

plt.cla()
plt.title("3D plot of Test Dataset : LDA Decision Boundary")
for name, label in [('Class1', 1), ('Class2', 2)]:
    ax_test.text3D(test_data[test_labels[:,0] == label, 0].mean(),
        test_data[test_labels[:,0] == label, 1].mean(),
        test_data[test_labels[:,0] == label, 2].mean(), name,
        horizontalalignment='center',
        bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Plot class 2 
ax_test.scatter(test_data[np.where(test_labels==2),0], test_data[np.where(test_labels==2),1], test_data[np.where(test_labels==2),2], c="g", cmap=plt.cm.spectral, edgecolor='k')

# Plot the decision boundary
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
zz = (-a*xx - b*yy - d)*1./c
ax_test.plot_surface(xx,yy,zz, color='yellow')

# Plot class 1 
ax_test.scatter(test_data[np.where(test_labels==1),0], test_data[np.where(test_labels==1),1], test_data[np.where(test_labels==1),2], c="r", cmap=plt.cm.spectral, edgecolor='k')

ax_test.w_xaxis.set_ticklabels([])
ax_test.w_yaxis.set_ticklabels([])
ax_test.w_zaxis.set_ticklabels([])


# Constructing decision boundary for the projected datasets
x_min, x_max = X_train.min() - 1, X_train.max() + 1
y_min, y_max = train_labels.min() - 1, train_labels.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = regression_object.predict(np.column_stack((np.ones(grid.shape[0]),grid[:, 0])))
probs = (np.argmax(probs, axis=1) + np.ones(probs.shape[0])).reshape(xx.shape)

# Projected Training set
plt.figure(5)
plt.title("Projected Train Dataset\nLinear Classification Boundary")
plt.xlabel("Projected direction")
plt.ylabel("Class")
plt.pcolormesh(xx, yy, probs, cmap="mesh")
plt.contour(xx, yy, probs, alpha=0.4, cmap="plasma")
plt.plot(X_train[train_labels[:,0] == 1],train_labels[train_labels[:,0] == 1],'o',c="r")
plt.plot(X_train[train_labels[:,0] == 2],train_labels[train_labels[:,0] == 2],'o',c="g")

#Projected Test set
plt.figure(6)
plt.title("Projected Test Dataset\nLinear Classification Boundary")
plt.xlabel("Projected direction")
plt.ylabel("Class")
plt.pcolormesh(xx, yy, probs, cmap="mesh")
plt.contour(xx, yy, probs, alpha=0.4, cmap="plasma")
plt.plot(X_test[test_labels[:,0] == 1],test_labels[test_labels[:,0] == 1],'o',c="r")
plt.plot(X_test[test_labels[:,0] == 2],test_labels[test_labels[:,0] == 2],'o',c="g")

# Predictions
x_min, x_max = plt.xlim()
x = np.arange(x_min, x_max, 0.1)
probs = regression_object.predict(np.column_stack((np.ones(x.shape[0]),x)))
probs = probs + 1

plt.figure(5)
plt.plot(x, probs[:,1], "b")

plt.figure(6)
plt.plot(x, probs[:,1], "b")

plt.show()