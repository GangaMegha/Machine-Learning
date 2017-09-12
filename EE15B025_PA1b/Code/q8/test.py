import numpy as np 

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score 

epochs = 20
hiddenLayerSize = 10
alpha = 0.1

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function

def softmax (x): return np.exp(x-np.amax(x))/np.sum(np.exp(x-np.amax(x)))

X_train = np.genfromtxt('../../Dataset/DS2-train-q8.csv', delimiter=",")
X_test = np.genfromtxt('../../Dataset/DS2-test-q8.csv', delimiter=",")

Y_train = X_train[:,-4:]
X_train = np.vstack((np.ones(X_train.shape[0]), X_train[:,:-4].T)).T
print X_train.shape
Y_test = X_test[:,-4:]
X_test = np.vstack((np.ones(X_test.shape[0]), X_test[:,:-4].T)).T

W1 = np.random.random((hiddenLayerSize, X_train.shape[1])) 
W2 = np.random.random((Y_train.shape[1], hiddenLayerSize+1))

W1, W2 = np.matrix(W1), np.matrix(W2)


X_train, Y_train = np.matrix(X_train), np.matrix(Y_train)
X_test, Y_test = np.matrix(X_test), np.matrix(Y_test)

training_loss = list()
test_loss = list()

num_iter = 5
for i in range(num_iter):
	r = np.random.choice(np.arange(1006),64)
	print r.shape


'''
for j in range(5):
	index = np.random.choice(np.arange(1006),64)
	train = X_train[index].T
	for i in range(epochs):
	# y  = forward_prop(X_train, W1, W2)
		a1 = W1 * train.T
		h1 = np.vstack((np.ones(a1.shape[1]),sigmoid(a1)))
		a2 = W2 * h1
		y_cap  = softmax(a2)
		
	# delta_w1, delta_w2 = back_prop()
		delta2 = y_cap - Y_train[index].T

		delta1 = np.multiply((W2.T)*delta2,np.multiply(h1,(1-h1)))
		
		# delta1 = np.multiply(((W2)*delta2.T).T, np.vstack((np.ones(len(h1)),np.multiply(h1,(1-h1)).T)).T)
		# delta1 = delta2*(W2.T)
		# temp = np.vstack((np.ones(h1.shape[0]),np.multiply(h1,1-h1).T)).T
		# print temp.shape	
		# delta1 = np.multiply(delta1 , temp)
		

		# print(delta1.shape)
		delta_w2 =  delta2 * h1.T
		# print((delta1.T).shape,train.shape)
		delta_w1 = delta1 * train.T 

		W1 = W1 - alpha * delta_w1[1:,:] 
		W2 = W2 - alpha * delta_w2

a1_cap = W1 * X_test.T
y_prediction = softmax(W2 * (np.vstack((np.ones(a1_cap.shape[1]),sigmoid(a1_cap))))).T

# print(y_prediction[np.where(Y_test==1)])
cross_entropy = -np.sum(np.log(y_prediction[np.where(Y_test==1)]))

print(cross_entropy)

# with open("Results.txt", "w") as text_file:
# 	for i in range(Y_test.shape[1]):
		
# 		accuracy = accuracy_score(np.array(Y_test[:,i]), np.array(y_prediction[:,i]))
# 		precision = average_precision_score(Y_test[:,i],y_prediction[:,i])
# 		recall = recall_score(Y_test[:,i], y_prediction[:,i], average='binary') 
# 		Fmeasure = f1_score(Y_test[:,i], y_prediction[:,i], average='binary') 

# 		text_file.write("\nAccuracy for class {} : {}".format(i+1, accuracy))
# 		text_file.write("\nPrecision for class {} : {}".format(i+1, precision))
# 		text_file.write("\nRecall for class {} : {}".format(i+1, recall))
# 		text_file.write("\nF-measure for class {} : {}".format(i+1, Fmeasure))

# clf_lr = LogisticRegression(penalty='l2')

# clf_lr.fit(X_train, Y_train)

# Y_cap	=	clf_lr.predict(X_test) # Predict the output for the test cases

with open("Results.txt", "w") as text_file:
	predicted_class = np.argmax(y_prediction, axis=1)
	k = y_prediction
	k[k<np.amax(y_prediction, axis =1)]=0
	k[k<np.amax(y_prediction, axis =1)]=1

	actual_class = np.argmax(Y_test, axis=1)
	print(predicted_class.shape, actual_class.shape)
	print(predicted_class)

	text_file.write("\nAccuracy : {}".format(accuracy_score(np.array(actual_class), np.array(predicted_class))))
	text_file.write("\nPrecision : {}".format(precision_score(np.array(Y_test), np.array(k), average='samples')))
	text_file.write("\nRecall : {}".format(recall_score(np.array(actual_class), np.array(predicted_class), average='micro') ))
	text_file.write("\nF-measure : {}".format(f1_score(np.array(actual_class), np.array(predicted_class), average='micro') ))

# with open("Coefficients.txt", "w") as text_file:
# 	text_file.write('Coefficients learned : {}'.format(clf_lr.coef_))