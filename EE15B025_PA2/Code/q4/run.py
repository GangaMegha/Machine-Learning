import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

def linear_kernel():

	i = 0
	accuracy = 0

	print("\n\n\nRunning SVM using Linear Kernel...................................")

	with open("Results/Result_linear", "w") as outfile:
		for c in penality :
			acr = []
			# Stratified k-fold cross-validation
			for train_indx, test_indx in skf.split(train_data, train_labels) :
				
				X_train, y_train = train_data[train_indx], train_labels[train_indx]
				X_test, y_test = train_data[test_indx], train_labels[test_indx]

				svm_obj = svm.SVC(C=c, kernel='linear', probability=False)
				svm_obj.fit(X_train, y_train)
							
				acr.append(svm_obj.score(X_test,y_test))

			svm_obj = svm.SVC(C=c, kernel='linear', probability=False)
			svm_obj.fit(train_data, train_labels)
			y_prediction = svm_obj.predict(test_data)

			# Write average accuracy and the classification report to reslut file
			outfile.write("\n\nFor C = {} : Average Accuracy = {}\n\n".format(c,  np.average(np.array(acr))))
			outfile.write(classification_report(test_labels, y_prediction, target_names=target_names))

			if accuracy <  np.average(np.array(acr)):
				accuracy =  np.average(np.array(acr))
				i = c
	print("\n\n")
	svm_obj = svm.SVC(C=i, kernel='linear', probability=False)
	svm_obj.fit(train_data, train_labels)
	y_prediction = svm_obj.predict(test_data)

# Save model
	pickle.dump(svm_obj, open("Models/svm model1.model", 'wb'))
	
	print("Result obtained using best model on test data :")
	print("\n\tC = {}\n\n\tAccuracy = {}\n".format(i, svm_obj.score(test_data,test_labels)))
	print(classification_report(test_labels, y_prediction, target_names=target_names))


def polynomial_kernel():

	i, j, k, l = 0, 0, 0, 0
	accuracy = 0

	print("\n\n\n\n\nRunning SVM using Polynomial Kernel...........................")

	with open("Results/Result_polynomial", "w") as outfile:
		for d in degree :
			for c in penality :
				for g in gamma :
					for r in coef0 :
						acr = []
						# Stratified k-fold cross-validation
						for train_indx, test_indx in skf.split(train_data, train_labels) :
							
							X_train, y_train = train_data[train_indx], train_labels[train_indx]
							X_test, y_test = train_data[test_indx], train_labels[test_indx]
							
							svm_obj = svm.SVC(C=c, kernel='poly', degree=d, gamma=g, coef0=r, probability=False)
							svm_obj.fit(X_train, y_train)
							
							acr.append(svm_obj.score(X_test,y_test))
						
						svm_obj = svm.SVC(C=c, kernel='poly', degree=d, gamma=g, coef0=r, probability=False)
						svm_obj.fit(train_data, train_labels)
						y_prediction = svm_obj.predict(test_data)
							
						# Write average accuracy and the classification report to reslut file
						outfile.write("\n\nFor Degree = {},\t C = {} ,\t Gamma = {} and\t Coef0 = {}\t : Average Accuracy = {}\n\n".format(d, c, g, r, np.average(np.array(acr))))
						outfile.write(classification_report(test_labels, y_prediction, target_names=target_names))
						
						if accuracy < np.average(np.array(acr)):
							accuracy = np.average(np.array(acr))
							i, j, k, l = d, g, r, c
	print("\n\n")
	svm_obj = svm.SVC(kernel='poly', degree=i, gamma=j, coef0=k, probability=False)
	svm_obj.fit(X_train, y_train)
	y_prediction = svm_obj.predict(test_data)
	
# Save model
	pickle.dump(svm_obj, open("Models/svm model2.model", 'wb'))

	print("Result obtained using best model on test data :")
	print("\n\tDegree = {}\n\tC = {}\n\tGamma = {}\n\tCoef0 = {}\n\n\tAccuracy = {}\n".format(i, l, j, k, svm_obj.score(X_test,y_test)))
	print(classification_report(test_labels, y_prediction, target_names=target_names))


def gaussean_kernel():
	
	i, j = 0, 0
	accuracy = 0

	print("\n\n\n\n\nRunning SVM using Gaussean Kernel...........................")

	with open("Results/Result_gaussean", "w") as outfile:
		for c in penality :
			for g in gamma :
				acr = []
				# Stratified k-fold cross-validation
				for train_indx, test_indx in skf.split(train_data, train_labels) :
					
					X_train, y_train = train_data[train_indx], train_labels[train_indx]
					X_test, y_test = train_data[test_indx], train_labels[test_indx]
					
					svm_obj = svm.SVC(C=c, kernel='rbf', gamma=g, probability=False)
					svm_obj.fit(X_train, y_train)
					
					acr.append(svm_obj.score(X_test,y_test))
				
				svm_obj = svm.SVC(C=c, kernel='rbf', gamma=g, probability=False)
				svm_obj.fit(train_data, train_labels)
				y_prediction = svm_obj.predict(test_data)
				
				# Write average accuracy and the classification report to reslut file
				outfile.write("\n\nFor C = {} and\t Gamma = {}\t : Average Accuracy = {}\n\n".format(c, g, np.average(np.array(acr))))
				outfile.write(classification_report(test_labels, y_prediction, target_names=target_names))
				
				if accuracy < np.average(np.array(acr)):
					accuracy = np.average(np.array(acr))
					i, j = c, g
	print("\n\n")
	svm_obj = svm.SVC(C=i, kernel='rbf', gamma=j, probability=False)
	svm_obj.fit(train_data, train_labels)
	y_prediction = svm_obj.predict(test_data)
	
# Save model
	pickle.dump(svm_obj, open("Models/svm model3.model", 'wb'))

	print("Result obtained using best model on test data :")
	print("\n\tC = {}\n\tGamma = {}\n\n\tAccuracy = {}\n".format(i, j, svm_obj.score(test_data, test_labels)))
	print(classification_report(test_labels, y_prediction, target_names=target_names))



def sigmoid_kernel():

	i, j, k = 0, 0, 0
	accuracy = 0

	print("\n\n\n\n\nRunning SVM using Sigmoid Kernel.............................")

	with open("Results/Result_sigmoid", "w") as outfile:
		for c in penality :
			for g in gamma :
				for r in coef0 :
					acr = []
					# Stratified k-fold cross-validation
					for train_indx, test_indx in skf.split(train_data, train_labels) :
						
						X_train, y_train = train_data[train_indx], train_labels[train_indx]
						X_test, y_test = train_data[test_indx], train_labels[test_indx]
					
						svm_obj = svm.SVC(C=c, kernel='sigmoid', gamma=g, coef0=r, probability=False)
						svm_obj.fit(X_train, y_train)

						acr.append(svm_obj.score(X_test,y_test))
					
					svm_obj = svm.SVC(C=c, kernel='sigmoid', gamma=g, coef0=r, probability=False)
					svm_obj.fit(train_data, train_labels)
					y_prediction = svm_obj.predict(test_data)
					
					# Write average accuracy and the classification report to reslut file
					outfile.write("\n\nFor C = {} and\t Gamma = {}\t Coef0 = {}\t : Average Accuracy = {}\n\n".format(c, g, r, np.average(np.array(acr))))
					outfile.write(classification_report(test_labels, y_prediction, target_names=target_names))
					
					if accuracy < np.average(np.array(acr)):
						accuracy = np.average(np.array(acr))
						i, j, k= c, g, r
	print("\n\n")
	svm_obj = svm.SVC(C=i, kernel='sigmoid', gamma=j, coef0=k, probability=False)
	svm_obj.fit(train_data, train_labels)
	y_prediction = svm_obj.predict(test_data)
	
# Save model
	pickle.dump(svm_obj, open("Models/svm model4.model", 'wb'))

	print("Result obtained using best model on test data :")
	print("\n\tC = {}\n\tGamma = {}\n\tCoef0 = {}\n\n\tAccuracy = {}\n".format(i, j, k, svm_obj.score(X_test,y_test)))
	print(classification_report(test_labels, y_prediction, target_names=target_names))

# Defining set of values for parameters
degree =  np.array([2])
penality = np.array([0.01, 0.0164285714286, 0.1, 3.31081081081, 4.71423423423, 94.7373684211, 500.0])
gamma = np.array([0.0111020408163, 0.021387755102, 0.1, 0.501351351351, 0.5125, 8.0, 10.5352631579])
coef0 = np.array([0.0, 0.01, 2.14285714286,7.15142857143])

target_names = ["coast", "forest", "inside city", "mountain"]

train_data, train_labels = load_svmlight_file("../../Dataset/DS2_train_libsvm")
test_data, test_labels = load_svmlight_file("../../Dataset/DS2_test_libsvm")

skf = StratifiedKFold(n_splits=10)

linear_kernel()
polynomial_kernel()
gaussean_kernel()
sigmoid_kernel()																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																	    