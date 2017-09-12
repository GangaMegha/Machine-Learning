import numpy as np 

from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, f1_score, recall_score 

epochs = 20
hiddenLayerSize = 20
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

W1 = np.random.random((X_train.shape[1], hiddenLayerSize)) 
W2 = np.random.random((hiddenLayerSize+1, Y_train.shape[1]))

W1, W2 = np.matrix(W1), np.matrix(W2)
# b1, b2 = np.random.random((1,hiddenLayerSize)), np.random.random(1,Y_train.shape[1])

X_train, Y_train = np.matrix(X_train), np.matrix(Y_train)
X_test, Y_test = np.matrix(X_test), np.matrix(Y_test)

for j in range(10):
	index = np.random.choice(np.arange(1006),64)
	train = X_train[index]
	for i in range(epochs):
	# y  = forward_prop(X_train, W1, W2)
		a1 = train * W1
		h1 = np.vstack((np.ones(a1.shape[0]),sigmoid(a1).T)).T
		a2 = h1 * W2
		y_cap  = softmax(a2)
		
	# delta_w1, delta_w2 = back_prop()
		delta2 = y_cap - Y_train[index]
		# delta1 = np.multiply(((W2)*delta2.T).T,np.multiply(h1,(1-h1)))
		delta1 = np.multiply(delta2*(W2.T),np.multiply(h1,(1-h1)))
		
		# delta1 = np.multiply(((W2)*delta2.T).T, np.vstack((np.ones(len(h1)),np.multiply(h1,(1-h1)).T)).T)
		# delta1 = delta2*(W2.T)
		# temp = np.vstack((np.ones(h1.shape[0]),np.multiply(h1,1-h1).T)).T
		# print temp.shape	
		# delta1 = np.multiply(delta1 , temp)
		

		# print(delta1.shape)
		delta_w2 =  delta2.T * h1
		delta_w1 = delta1.T * train 
		# delta_w2 =  delta_w2 + delta2.T * h1
		# delta_w1 = delta_w1 + delta1.T * train 

		W1 = W1 - alpha * delta_w1.T[:,1:] 
		W2 = W2 - alpha * delta_w2.T

a1_cap = X_test*W1
y_prediction = softmax((np.vstack((np.ones(a1_cap.shape[0]),sigmoid(a1_cap).T)).T)*W2)

print(y_prediction[np.where(Y_test==1)])
cross_validation = -np.sum(np.log(y_prediction[np.where(Y_test==1)]))

print(cross_validation)

# clf_lr = LogisticRegression(penalty='l2')

# clf_lr.fit(X_train, Y_train)

# Y_cap	=	clf_lr.predict(X_test) # Predict the output for the test cases
a = []
p = []
r = []
f = []
with open("Results.txt", "w") as text_file:
	for i in range(Y_test.shape[1]):
		
		accuracy = accuracy_score(np.array(Y_test), np.array(Y_cap))
		precision = average_precision_score(Y_test,Y_cap)
		recall = recall_score(Y_test, Y_cap, average='binary') 
		Fmeasure = f1_score(Y_test, Y_cap, average='binary') 

		text_file.write("\nAccuracy for class {} : {}".format(i+1, accuracy))
		text_file.write("\nPrecision for class {} : {}".format(i+1, precision))
		text_file.write("\nRecall for class {} : {}".format(i+1, recall))
		text_file.write("\nF-measure for class {} : {}".format(i+1, Fmeasure))

		a.append(accuracy)
		p.append(precision)
		r.append(recall)
		f.append(Fmeasure)

	text_file.write("\nAverage Accuracy : {}".format(np.sum(a)/(Y_test.shape[1])))
	text_file.write("\nAverage Precision : {}".format(np.sum(p)/(Y_test.shape[1])))
	text_file.write("\nAverage Recall : {}".format(np.sum(r)/(Y_test.shape[1])))
	text_file.write("\nAverage F-measure : {}".format(np.sum(f)/(Y_test.shape[1])))


# with open("Coefficients.txt", "w") as text_file:
# 	text_file.write('Coefficients learned : {} \n\n\t {}'.format(W1,W2))