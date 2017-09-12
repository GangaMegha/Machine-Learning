import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, f1_score, recall_score 

X_train = np.genfromtxt('../../Dataset/DS2-train.csv', delimiter=",")
X_test = np.genfromtxt('../../Dataset/DS2-test.csv', delimiter=",")

Y_train = X_train[:,-1]
X_train = np.vstack((np.ones(X_train.shape[0]), X_train[:,:-1].T)).T

Y_test = X_test[:,-1]
X_test = np.vstack((np.ones(X_test.shape[0]), X_test[:,:-1].T)).T

clf_lr = LogisticRegression(penalty='l2')

clf_lr.fit(X_train, Y_train)

Y_cap	=	clf_lr.predict(X_test) # Predict the output for the test cases

with open("Results.txt", "w") as text_file:
	text_file.write("\nAccuracy : {}".format(accuracy_score(np.array(Y_test), np.array(Y_cap))))
	text_file.write("\nPrecision : {}".format(average_precision_score(Y_test,Y_cap)))
	text_file.write("\nRecall : {}".format(recall_score(Y_test, Y_cap, average='binary') ))
	text_file.write("\nF-measure : {}".format(f1_score(Y_test, Y_cap, average='binary') ))

with open("Coefficients.txt", "w") as text_file:
	text_file.write('Coefficients learned : {}'.format(clf_lr.coef_))