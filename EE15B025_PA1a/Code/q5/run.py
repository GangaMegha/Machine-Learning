import numpy as np 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score

dataset = np.genfromtxt('../q4/CandC_modified.csv', delimiter=",") #reading modified imputed data from the csv file

Residual_error = 0 # For holding the average error

m = 0 #storing the number of samples.

for i in range(5) : # For iterating over the 5 80:20% splits of the dataset

	np.savetxt("../../Dataset/CandC-train{}.csv".format(i+1), dataset[i*399:(i+1)*399-80,:], delimiter=',') # Writing the train dataset 'i' into a csv file
	np.savetxt("../../Dataset/CandC-test{}.csv".format(i+1), dataset[(i+1)*399-80:(i+1)*399,:], delimiter=',') # Writing the test dataset 'i' into a csv file

	X_train = np.vstack((np.ones(319),dataset[i*399:(i+1)*399-80,:-1].transpose())).transpose()  # appending a vector of 1s as the initial column
	Y_train = dataset[i*399:(i+1)*399-80,-1] # Last column contains the desired output y

	try : # Since one of the test sets have only 79 datapoints
		X_test = np.vstack((np.ones(80),dataset[(i+1)*399-80:(i+1)*399,:-1].transpose())).transpose() # appending a vector of 1s as the initial column
		Y_test = dataset[(i+1)*399-80:(i+1)*399,-1] # Last column contains the desired output y
	except :
		X_test = np.vstack((np.ones(79),dataset[(i+1)*399-80:(i+1)*399,:-1].transpose())).transpose() # appending a vector of 1s as the initial column
		Y_test = dataset[(i+1)*399-80:(i+1)*399,-1] # Last column contains the desired output y

	m = Y_test.shape[0]

	regr_object = linear_model.LinearRegression() # Object for linear regression

	regr_object.fit(X_train, Y_train) # Fit Linear regression model

	Y_cap = regr_object.predict(X_test) # Predict using the linear model

	with open("Coefficients_{}.txt".format(i+1), "w") as text_file:
		text_file.write('Coefficients learned : {}'.format(regr_object.coef_)) # saving the learned coefficients

	Residual_error += m*mean_squared_error(Y_test, Y_cap) # Mean squared error regression loss

Residual_error /= 5 # Taking average of the errors

print("\n\nAverage Residual Error : {}\n\n".format(Residual_error))

# Average Residual Error : 2.60444840251