import numpy as np 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, f1_score, recall_score 

X = np.genfromtxt('../../Dataset/DS1-train.csv', delimiter=",") # reading train data from the csv file

Y 	=	X[:,-2:]	# Y matrix : one hot encoded
X 	= 	X[:,:-2]	# X matrix

X_test = np.genfromtxt('../../Dataset/DS1-test.csv', delimiter=",") # reading test data from the csv file

np.random.shuffle(X_test)	#shuffling the [X,Y]

Y_test 	=	X_test[:,-2:]	# Y matrix : one hot encoded
X_test 	= 	X_test[:,:-2]	# X matrix

regression_object	=	linear_model.LinearRegression()	# Create linear regression object

regression_object.fit(X, Y)	# Train the model using the training sets

Y_cap	=	regression_object.predict(X_test) # Predict the output for the test cases

predicted_class = np.argmax(Y_cap, axis=1) # Take the column corresponding to the predicted maximum for each test case

with open("Results.txt", "w") as text_file:
	text_file.write("\nAccuracy : {}".format(accuracy_score(np.array(Y_test[:,1]), np.array(predicted_class))))
	text_file.write("\nPrecision : {}".format(average_precision_score(Y_test[:,1],predicted_class)))
	text_file.write("\nRecall : {}".format(recall_score(Y_test[:,1], predicted_class, average='binary') ))
	text_file.write("\nF-measure : {}".format(f1_score(Y_test[:,1], predicted_class, average='binary') ))

with open("Coefficients.txt", "w") as text_file:
	text_file.write('Coefficients learned : {}'.format(regression_object.coef_))