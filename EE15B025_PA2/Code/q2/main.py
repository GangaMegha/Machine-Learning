import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score, classification_report

# Read data from files
train_data = np.array(list(csv.reader(open("../../Dataset/DS3/train.csv", "rb"), delimiter=","))).astype("float")
train_labels = np.array(list(csv.reader(open("../../Dataset/DS3/train_labels.csv", "rb"), delimiter=","))).astype("int")

test_data = np.array(list(csv.reader(open("../../Dataset/DS3/test.csv", "rb"), delimiter=","))).astype("float")
test_labels = np.array(list(csv.reader(open("../../Dataset/DS3/test_labels.csv", "rb"), delimiter=","))).astype("int")

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(train_data, train_labels)
X_train = lda.transform(train_data)
Y_train = np.zeros((X_train.shape[0],2))
Y_train[np.where(train_labels==1)[0]] = [1, 0]
Y_train[np.where(train_labels==2)[0]] = [0, 1]

X_test = lda.transform(test_data)
Y_test = np.zeros((X_test.shape[0],2))
Y_test[np.where(test_labels==1)[0]] = [1, 0]
Y_test[np.where(test_labels==2)[0]] = [0, 1]

# Prediction by LDA
Y_cap1	=	lda.predict(test_data) # Predict the output for the test cases

#Prediction by Linear Regression
regression_object	=	linear_model.LinearRegression()	# Create linear regression object

X_train_intercept = np.column_stack((np.ones(X_train.shape[0]),X_train))
X_test_intercept = np.column_stack((np.ones(X_test.shape[0]),X_test))

regression_object.fit(X_train_intercept, Y_train)	# Train the model using the training sets

Y_cap2	=	regression_object.predict(X_test_intercept) # Predict the output for the test cases
Y_cap2 = np.argmax(Y_cap2, axis=1) + np.ones(Y_cap2.shape[0])# Take the column corresponding to the predicted maximum for each test case

# Write results to file
with open("Results.txt", "w") as text_file:
	text_file.write("\nPrediction by LDA :")
	text_file.write("\n\nPrecision for Class 1 : {} \t\t Precision for Class 2 : {}".format(precision_score(test_labels,Y_cap1, average=None)[0], precision_score(test_labels,Y_cap1, average=None)[1] ))
	text_file.write("\n\nRecall for Class 1 : {} \t\t\t Recall for Class 2 : {}".format(recall_score(test_labels, Y_cap1, average=None)[0], recall_score(test_labels, Y_cap1, average=None)[1] ))
	text_file.write("\n\nF-measure for Class 1 : {} \t\t F-measure for Class 2 : {}".format(f1_score(test_labels, Y_cap1, average=None)[0], f1_score(test_labels, Y_cap1, average=None)[1] ))
	text_file.write("\n\n\n\nOverall Accuracy : {}".format(lda.score(test_data, np.array(test_labels))))

	text_file.write("\n\n\n\n\n\nPrediction by Linear Regression (after feature extraction) :")
	text_file.write("\n\nPrecision for Class 1 : {} \t\t Precision for Class 2 : {}".format(precision_score(test_labels,Y_cap2, average=None)[0], precision_score(test_labels,Y_cap2, average=None)[1] ))
	text_file.write("\n\nRecall for Class 1 : {} \t\t\t Recall for Class 2 : {}".format(recall_score(test_labels, Y_cap2, average=None)[0], recall_score(test_labels, Y_cap2, average=None)[1] ))
	text_file.write("\n\nF-measure for Class 1 : {} \t\t F-measure for Class 2 : {}".format(f1_score(test_labels, Y_cap2, average=None)[0], f1_score(test_labels, Y_cap2, average=None)[1] ))
	text_file.write("\n\n\n\nOverall Accuracy : {}".format(lda.score(test_data, np.array(test_labels))))

# Write Coefficients to file
with open("Coefficients.txt", "w") as text_file:
	text_file.write('\nCoefficients learned by LDA: \n\t\t{}'.format(lda.coef_))
	text_file.write('\n\nCoefficients learned by Linear regression after feature extraction: \n\t\t{}'.format(regression_object.coef_))