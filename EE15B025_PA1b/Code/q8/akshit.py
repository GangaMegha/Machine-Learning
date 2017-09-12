import numpy as np 
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, f1_score, recall_score 
import matplotlib.pyplot as plt

epochs = 500
hiddenLayerSize = 200
alpha = 0.01
lbda = 0.1

def sigmoid (x): 
	return 1/(1 + np.exp(-x))      

def sigmoidGradient(x):
	return np.multiply(sigmoid(x), 1 - sigmoid(x))

def softmax (x): 
	return np.exp(x-np.amax(x,axis=1))/np.sum(np.exp(x-np.amax(x,axis=1)),axis=1)

X_train = np.genfromtxt('../../Dataset/DS2-train-q8.csv', delimiter=",")
X_test = np.genfromtxt('../../Dataset/DS2-test-q8.csv', delimiter=",")

Y_train = X_train[:,-4:]
X_train = X_train[:,:-4]

Y_test = X_test[:,-4:]
X_test = X_test[:,:-4]

#W1 = np.random.random((X_train.shape[1] + 1, hiddenLayerSize)) 
#W2 = np.random.random((hiddenLayerSize+1, Y_train.shape[1]))

W1 = np.random.randn(hiddenLayerSize , X_train.shape[1] + 1)
W2 = np.random.randn(Y_train.shape[1], hiddenLayerSize + 1)

W1, W2 = np.matrix(W1), np.matrix(W2)

X_train, Y_train = np.matrix(X_train), np.matrix(Y_train)
X_test, Y_test = np.matrix(X_test), np.matrix(Y_test)

training_loss = list()

num_iter = 5;
for i in range(num_iter):
	r = np.random.choice(np.arange(1006),64)
	x_train = X_train[r]
	y_train = Y_train[r]
	# print y_train
	# print x_train

	# Calculating the Gradients
	m = x_train.shape[0]
	a1 = np.column_stack((np.ones(m),x_train))
	
	z2 = a1 * W1.T
	#print z2
	a2 = np.column_stack((np.ones(z2.shape[0]),sigmoid(z2)))
	#print a2
	z3 = a2 * W2.T
	#print z3
	y_hat = softmax(z3)
	

	# Calculating the cost function loss
	L = -np.sum(np.sum(np.multiply(y_train,np.log(y_hat + 1e-11))))
	L = L/m;
	reg = (lbda/2*m) * (np.sum(np.sum(np.multiply(W1[:,1:],W1[:,1:]))) + np.sum(np.sum(np.multiply(W2[:,1:],W2[:,1:]))))
	L = L + reg;
	training_loss.append(L)


	# Back Progogation
	d3 = y_hat - y_train
	#print d3
	# print y_hat
	# print y_train
	d2 = (d3 * W2)
	temp = np.column_stack((np.ones(z2.shape[0]),sigmoidGradient(z2)))
	d2 = np.multiply(d2,temp)

	D1 = d2[:,1:].T * a1;
	D1 = D1/m;
	D1[:,1:] = D1[:,1:] + (lbda)*W1[:,1:]
	
	D2 = d3.T * a2
	D2 = D2/m;
	D2[:,1:] = D2[:,1:] + (lbda/m)*W2[:,1:]
	#print D1
	#print D2
	#print D2
	# Weight Update
	W1 = W1 - alpha * D1;
	W2 = W2 - alpha * D2;

x = np.arange(10000)

# plt.plot(x,training_loss)
# plt.show()

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


print("\nAccuracy : {}".format(accuracy_score(np.array(actual_class), np.array(predicted_class))))
'''
for j in range(20):
	index = np.random.choice(np.arange(1006),64)
	train = X_train[index]
	m = 64
	for i in range(epochs):
	# y  = forward_prop(X_train, W1, W2)
		a1 = train * W1
		h1 = np.vstack((np.ones(a1.shape[0]),sigmoid(a1).T)).T
		a2 = h1 * W2
		y_cap  = softmax(a2)
		
		training_cross_entropy = -np.sum(np.sum(np.multiply(Y_train[index],np.log(y_cap +1e-7))))
		reg = (lbda/2*m) * (np.sum(np.sum(np.multiply(W1[:,1:],W1[:,1:]))) + np.sum(np.sum(np.multiply(W2[:,1:],W2[:,1:]))))
		print training_cross_entropy/len(X_train)  + reg
	# delta_w1, delta_w2 = back_prop()
		delta2 = y_cap - Y_train[index]

		delta1 = np.multiply(delta2*(W2.T),np.multiply(h1,(1-h1)))
		
		# delta1 = np.multiply(((W2)*delta2.T).T, np.vstack((np.ones(len(h1)),np.multiply(h1,(1-h1)).T)).T)
		# delta1 = delta2*(W2.T)
		# temp = np.vstack((np.ones(h1.shape[0]),np.multiply(h1,1-h1).T)).T
		# print temp.shape	
		# delta1 = np.multiply(delta1 , temp)
		

		# print(delta1.shape)
		delta_w2 =  delta2.T * h1
		delta_w2 = delta_w2 / len(X_train)
		# print((delta1.T).shape,train.shape)
		delta_w1 = delta1.T * train 
		delta_w1 = delta_w1 / len(X_train)
		print delta_w1.shape
		delta_w1[:,1:] = delta_w1[:,1:] + (lbda/m) * W1.T[:,1:]
		delta_w2[:,1:] = delta_w2[:,1:] + (lbda/m) * W2.T[:,1:]

		W1 = W1 - alpha * delta_w1.T[:,1:] 
		W2 = W2 - alpha * delta_w2.T
		

a1_cap = X_test*W1
y_prediction = softmax((np.vstack((np.ones(a1_cap.shape[0]),sigmoid(a1_cap).T)).T)*W2)

#print(y_prediction[np.where(Y_test==1)])
cross_entropy = -np.sum(np.sum(np.multiply(Y_test,np.log(y_prediction + 1e-7))))
print cross_entropy / Y_test.shape[0]

#print(cross_validation)

# clf_lr = LogisticRegression(penalty='l2')

# clf_lr.fit(X_train, Y_train)

# Y_cap	=	clf_lr.predict(X_test) # Predict the output for the test cases

# with open("Results.txt", "w") as text_file:
# 	text_file.write("\nAccuracy : {}".format(accuracy_score(np.array(Y_test), np.array(Y_cap))))
# 	text_file.write("\nPrecision : {}".format(average_precision_score(Y_test,Y_cap)))
# 	text_file.write("\nRecall : {}".format(recall_score(Y_test, Y_cap, average='binary') ))
# 	text_file.write("\nF-measure : {}".format(f1_score(Y_test, Y_cap, average='binary') ))

# with open("Coefficients.txt", "w") as text_file:
# 	text_file.write('Coefficients learned : {}'.format(clf_lr.coef_))
'''