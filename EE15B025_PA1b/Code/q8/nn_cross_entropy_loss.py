import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv

train = np.array(list(csv.reader(open("DS2_train_norm.csv", "rb"), delimiter=","))).astype("float")
X_train = train[:,:-1]
X_train = preprocessing.scale(X_train.T).T
y_train = train[:,-1]
y_train = y_train.astype(int)
Y_train = np.eye(4)[y_train]

test = np.array(list(csv.reader(open("DS2_test_norm.csv", "rb"), delimiter=","))).astype("float")
X_test = test[:,:-1]
X_test = preprocessing.scale(X_test.T).T
y_test = test[:,-1]
y_test = y_test.astype(int)
Y_test = np.eye(4)[y_test]

hiddenLayerSize = 1000
alpha = 0.1

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def sigmoidGradient(x):
	return np.multiply(sigmoid(x), 1 - sigmoid(x))

def softmax (x):
	return np.exp(x-np.amax(x,axis=1))/np.sum(np.exp(x-np.amax(x,axis=1)),axis=1)

lambdas = [0, 0.01, 0.1 , 1, 10, 100];

for lbda in lambdas:
	W1 = np.random.randn(hiddenLayerSize , X_train.shape[1] + 1)
	W2 = np.random.randn(Y_train.shape[1], hiddenLayerSize + 1)
	W1, W2 = np.matrix(W1), np.matrix(W2)
	X_train, Y_train = np.matrix(X_train), np.matrix(Y_train)
	X_test, Y_test = np.matrix(X_test), np.matrix(Y_test)
	
	print "Running for lambda = %g" %(lbda)
	
	num_iter = 10000
	Loss = []
	
	for i in range(num_iter):
		r = np.random.choice(np.arange(1006),64)
		x_train = X_train[r]
		y_train = Y_train[r]

        # Calculating the gradients
		m = x_train.shape[0]
		a1 = np.column_stack((np.ones(m),x_train))
		z2 = a1 * W1.T
		a2 = np.column_stack((np.ones(z2.shape[0]),sigmoid(z2)))
		z3 = a2 * W2.T
		y_hat = softmax(z3)


		# Calculating the cost function loss
		L = -np.sum(np.sum(np.multiply(y_train,np.log(y_hat + 1e-11))))
		L = L/m;
		reg = (lbda/2*m) * (np.sum(np.sum(np.multiply(W1[:,1:],W1[:,1:]))) + np.sum(np.sum(np.multiply(W2[:,1:],W2[:,1:]))))
		L = L + reg;
		Loss.append(L)

		# Back Progogation
		d3 = y_hat - y_train
		d2 = (d3 * W2)
		temp = np.column_stack((np.ones(z2.shape[0]),sigmoidGradient(z2)))
		d2 = np.multiply(d2,temp)

		D1 = d2[:,1:].T * a1;
		D1 = D1/m;
		D1[:,1:] = D1[:,1:] + (lbda/m)*W1[:,1:]

		D2 = d3.T * a2
		D2 = D2/m;
		D2[:,1:] = D2[:,1:] + (lbda/m)*W2[:,1:]
		
		# Weight Update
		W1 = W1 - alpha * D1;
		W2 = W2 - alpha * D2;

	# Testing on the test data set
	x_test = X_test
	y_test = Y_test
	m = x_test.shape[0]
	a1 = np.column_stack((np.ones(m),x_test))
	z2 = a1 * W1.T
	a2 = np.column_stack((np.ones(z2.shape[0]),sigmoid(z2)))
	z3 = a2 * W2.T
	y_hat = softmax(z3)
	predicted_class = np.argmax(y_hat, axis=1)
	actual_class = np.argmax(y_test, axis=1)
	plt.plot(np.arange(num_iter),Loss)
	plt.show()	
	print("\nAccuracy : {}\n".format(accuracy_score(np.array(actual_class), np.array(predicted_class))))
	
	precision = precision_score(actual_class, predicted_class, average=None)
	print("Precision : Mountain = {}, Forest = {}, Coast = {}, Insidecity = {}\n".format(precision[0],precision[1],precision[2],precision[3]))
	
	Fmeasure = f1_score(actual_class, predicted_class, average=None)
	print("F-measure : Mountain = {}, Forest = {}, Coast = {}, Insidecity = {}\n".format(Fmeasure[0],Fmeasure[1],Fmeasure[2],Fmeasure[3]))

	Recall = recall_score(actual_class, predicted_class, average=None)
	print("Recall : Mountain = {}, Forest = {}, Coast = {}, Insidecity = {}\n".format(Recall[0],Recall[1],Recall[2],Recall[3]))
