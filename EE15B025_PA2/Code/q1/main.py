import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score, classification_report

# Read in data from the csv files
train_data = np.array(list(csv.reader(open("../../Dataset/DS3/train.csv", "rb"), delimiter=","))).astype("float")
train_labels = np.array(list(csv.reader(open("../../Dataset/DS3/train_labels.csv", "rb"), delimiter=","))).astype("int")

test_data = np.array(list(csv.reader(open("../../Dataset/DS3/test.csv", "rb"), delimiter=","))).astype("float")
test_labels = np.array(list(csv.reader(open("../../Dataset/DS3/test_labels.csv", "rb"), delimiter=","))).astype("int")

# Fitting and transformation for PCA
pca = decomposition.PCA(n_components=1)
pca.fit(train_data)
X_train = pca.transform(train_data)
Y_train = np.zeros((X_train.shape[0],2))
Y_train[np.where(train_labels==1)[0]] = [1, 0]
Y_train[np.where(train_labels==2)[0]] = [0, 1]

X_test = pca.transform(test_data)
Y_test = np.zeros((X_test.shape[0],2))
Y_test[np.where(test_labels==1)[0]] = [1, 0]
Y_test[np.where(test_labels==2)[0]] = [0, 1]

# Suffling the training data
X = np.column_stack((X_train, Y_train[:,0]))
X = np.column_stack((X, Y_train[:,1]))
np.random.shuffle(X)
X_train = X[:,:-2]
Y_train = X[:,-2:]

regression_object	=	linear_model.LinearRegression()	# Create linear regression object

X_train_intercept = np.column_stack((np.ones(X_train.shape[0]),X_train))
X_test_intercept = np.column_stack((np.ones(X_test.shape[0]),X_test))

regression_object.fit(X_train_intercept, Y_train)	# Train the model using the training sets

Y_cap	=	regression_object.predict(X_test_intercept) # Predict the output for the test cases
Y_cap = np.argmax(Y_cap, axis=1) + np.ones(Y_cap.shape[0])# Take the column corresponding to the predicted maximum for each test case

# Writing the results onto the result file
with open("Results.txt", "w") as text_file:
	text_file.write("\n\nPrecision for Class 1 : {} \t\t Precision for Class 2 : {}".format(precision_score(test_labels,Y_cap, average=None)[0], precision_score(test_labels,Y_cap, average=None)[1] ))
	text_file.write("\n\nRecall for Class 1 : {} \t\t\t\t\t Recall for Class 2 : {}".format(recall_score(test_labels, Y_cap, average=None)[0], recall_score(test_labels, Y_cap, average=None)[1] ))
	text_file.write("\n\nF-measure for Class 1 : {} \t\t F-measure for Class 2 : {}".format(f1_score(test_labels, Y_cap, average=None)[0], f1_score(test_labels, Y_cap, average=None)[1] ))
	text_file.write("\n\n\n\n\n\nOverall Accuracy : {}".format(accuracy_score(np.array(test_labels), np.array(Y_cap))))

# Writing the coefficients
with open("Coefficients.txt", "w") as text_file:
	text_file.write('Coefficients learned : {}'.format(regression_object.coef_))
