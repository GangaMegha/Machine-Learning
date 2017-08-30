import numpy as np 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, f1_score, recall_score #, precision_score
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt('../../Dataset/DS1-train.csv', delimiter=",") # reading train data from the csv file

Y 	=	X[:,-2:]	# Y matrix : one hot encoded
X 	= 	X[:,:-2]	# X matrix
# print(X.shape,Y.shape) 
X_test = np.genfromtxt('../../Dataset/DS1-test.csv', delimiter=",") # reading test data from the csv file

Y_test 	=	X_test[:,-2:]	# Y matrix : one hot encoded
X_test 	= 	X_test[:,:-2]	# X matrix

k=0 # Denotes the number of neighbours used for classification

accuracy = [] # List for storing accuracy values correponding to each k
precision = [] # List for storing precision values correponding to each k
recall = [] # List for storing recall values correponding to each k
fmeasure = [] # List for storing F-measure values correponding to each k

text_file = open("Results.txt", "w")

for i in range(250) :
	
	k+=1 # increment the neighbours by 1

	text_file.write("\nk : {}".format(k))

	knn_object	=	KNeighborsClassifier(n_neighbors=k)	# Create linear regression object

	knn_object.fit(X, Y)	# Train the model using the training sets

	Y_cap	=	knn_object.predict(X_test) # Predict the output for the test cases

	predicted_class = np.argmax(Y_cap, axis=1) # Take the column corresponding to the predicted maximum for each test case

	# We can compare predicted class with the second column of Y_test (since Y_test is one hot encoded and it's a binary classification problem)

	text_file.write("\n\tAccuracy : {}".format(accuracy_score(np.array(Y_test[:,1]), np.array(predicted_class))))
	text_file.write("\n\tPrecision : {}".format(average_precision_score(Y_test[:,1], predicted_class)))
	text_file.write("\n\tRecall : {}".format(recall_score(Y_test[:,1], predicted_class, average='binary') ))
	text_file.write("\n\tF-measure : {}".format(f1_score(Y_test[:,1], predicted_class, average='binary') ))

	accuracy.append(accuracy_score(np.array(Y_test[:,1]), np.array(predicted_class))) 	# Appending the accuracy
	precision.append(average_precision_score(Y_test[:,1], predicted_class))				# Appending the precision 
	recall.append(recall_score(Y_test[:,1], predicted_class, average='binary'))			# Appending the recall 
	fmeasure.append(f1_score(Y_test[:,1], predicted_class, average='binary'))			# Appending the F-measure

text_file.close()

with open("Best_results.txt", "w") as text_file:
    text_file.write("\nBest Accuracy = {} for k = {}".format(np.max(accuracy), np.argmax(accuracy)))
    text_file.write("\nBest Precision = {} for k = {}".format(np.max(precision), np.argmax(precision)))
    text_file.write("\nBest Recall = {} for k = {}".format(np.max(recall), np.argmax(recall)))
    text_file.write("\nBest F-measure = {} for k = {}".format(np.max(fmeasure), np.argmax(fmeasure)))

print("Best Accuracy = {} for k = {}".format(np.max(accuracy),np.argmax(accuracy)))
print("Best Precision = {} for k = {}".format(np.max(precision),np.argmax(precision)))
print("Best Recall = {} for k = {}".format(np.max(recall), np.argmax(recall)))
print("Best F-measure = {} for k = {}".format(np.max(fmeasure), np.argmax(fmeasure)))
