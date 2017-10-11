import numpy as np 
import csv

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score 

Y_cap = np.genfromtxt('l1_logreg-0.8.2-i686-pc-linux-gnu/test_prediction', delimiter=",")
Y_test = np.genfromtxt('l1_logreg-0.8.2-i686-pc-linux-gnu/Test_labels', delimiter=",")

print("\nPrecision for class Forest : {} and Mountain : {}".format(precision_score(Y_test,Y_cap, average=None)[0], precision_score(Y_test,Y_cap, average=None)[1]))
print("\nRecall for class Forest : {} and Mountain : {}".format(recall_score(Y_test,Y_cap, average=None)[0], recall_score(Y_test,Y_cap, average=None)[1]))
print("\nF-measure for class Forest : {} and Mountain : {}".format(f1_score(Y_test,Y_cap, average=None)[0], f1_score(Y_test,Y_cap, average=None)[1]))

print("\nOverall Accuracy : {}".format(accuracy_score(np.array(Y_test), np.array(Y_cap))))

print("\nCorrect Labels : {}".format(Y_test))
print("\nPredicted Labels : {}".format(Y_cap))