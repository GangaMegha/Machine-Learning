import numpy as np 
from sklearn import preprocessing

X_train = np.genfromtxt('../../Dataset/Train_features', delimiter = ",")
Y_train = np.genfromtxt('../../Dataset/Train_labels', delimiter = ",")
X_test = np.genfromtxt('../../Dataset/Test_features', delimiter = ",")
Y_test = np.genfromtxt('../../Dataset/Test_labels', delimiter = ",")

X_train = np.reshape(X_train[2:], (96,1006)).T
Y_train = np.reshape(Y_train[2:], (4, 1006))

X_test = np.reshape(X_test[2:], (96,80)).T
Y_test = np.reshape(Y_test[2:], (4,80))

X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

Train = np.vstack((X_train.T,Y_train)).T 
Test = np.vstack((X_test.T,Y_test)).T

np.random.shuffle(Train)
np.random.shuffle(Test)

np.savetxt("../../Dataset/DS2-train-q8.csv", Train, delimiter=',') # Writing the output into a csv file
np.savetxt("../../Dataset/DS2-test-q8.csv", Test, delimiter=',') # Writing the output into a csv file