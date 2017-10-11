import numpy as np 

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score 

X_train = np.genfromtxt('../../Dataset/DS2-train.csv', delimiter=",")
X_test = np.genfromtxt('../../Dataset/DS2-test.csv', delimiter=",")

Y_train = X_train[:,-1]
X_train = np.vstack((np.ones(X_train.shape[0]), X_train[:,:-1].T)).T

Y_test = X_test[:,-1]
X_test = np.vstack((np.ones(X_test.shape[0]), X_test[:,:-1].T)).T

clf_lr = Ridge(alpha=0.01)

clf_lr.fit(X_train, Y_train)

Y_cap	=	clf_lr.predict(X_test) # Predict the output for the test cases

Y_cap[np.where(Y_cap<0)] = -1
Y_cap[np.where(Y_cap>0)] = 1

print("\nPrecision for class Forest : {} and Mountain : {}".format(precision_score(Y_test,Y_cap, average=None)[0], precision_score(Y_test,Y_cap, average=None)[1]))
print("\nRecall for class Forest : {} and Mountain : {}".format(recall_score(Y_test,Y_cap, average=None)[0], recall_score(Y_test,Y_cap, average=None)[1]))
print("\nF-measure for class Forest : {} and Mountain : {}".format(f1_score(Y_test,Y_cap, average=None)[0], f1_score(Y_test,Y_cap, average=None)[1]))

print("\nOverall Accuracy : {}".format(accuracy_score(np.array(Y_test), np.array(Y_cap))))

print("\nCorrect Labels : {}".format(Y_test))
print("\nPredicted Labels : {}".format(Y_cap))