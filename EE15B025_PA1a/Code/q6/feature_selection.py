######################### Using reduced features ####################################

import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from ridge_lamda import optimal_lamda # getting the optimum lamda from ridge_lamda.py 

print("\nRunning code for feature selection using ridge regression.........\n\n")
print("Output written onto Feature_selection.txt\n\n")

error = [] # List for storing the error values

text_file = open("Feature_selection.txt", "w") # Contains the k values used, average residual error obtained and the no. of features considered

text_file.write("The features are selected based on the criteria that : |coefficient or weights| > k*standard_deviation of coefficients(learned) from mean of the coeficients(learned).")
text_file.write("\nThe results obtained are as follows : \n")

k = 0 # k is a scaling factor for standard deviation while estimating the prominant features

m = 0 #storing the number of samples.

for j in range(20): # Iterate over different scaling values. The number of iterations and k values have been chosen after a lot of trials 

	Residual_error = 0 # Holds the average residual error for each lamda value

	for i in range(5) : # Iterating over the different datasets

		Train = np.genfromtxt("../../Dataset/CandC-train{}.csv".format(i+1), delimiter=",") # Reading the train data from a csv file
		Test = np.genfromtxt("../../Dataset/CandC-test{}.csv".format(i+1), delimiter=",") # Reading the test data from a csv file

		X_train = np.vstack(( np.ones(319), Train[:,:-1].transpose() )).transpose() # appending a vector of 1s as the initial column
		Y_train = Train[:,-1:] # Last column contains the desired output y

		try : # Since one of the test sets have only 79 datapoints
			X_test = np.vstack(( np.ones(80), Test[:,:-1].transpose() )).transpose() # appending a vector of 1s as the initial column
			Y_test = Test[:,-1:] # Last column contains the desired output y
		except :
			X_test = np.vstack(( np.ones(79), Test[:,:-1].transpose() )).transpose() # appending a vector of 1s as the initial column
			Y_test = Test[:,-1:] # Last column contains the desired output y

		ridge_object = Ridge(alpha=optimal_lamda) # Ridge object initialised with optimal lamda obtained in ridge_lamda

		ridge_object.fit(X_train,Y_train) # Fit Ridge regression model

		required_columns = np.where( (np.absolute(ridge_object.coef_ - np.mean(ridge_object.coef_))>k*np.std(ridge_object.coef_)) )[1] # getting column numbers of important features
		
		X_train = (X_train)[:,required_columns] # getting the prominant features
			
		X_test = (X_test)[:,required_columns] # getting the prominant features

		ridge_object = Ridge(alpha=optimal_lamda) # Ridge object initialised with optimal lamda obtained in ridge_lamda

		ridge_object.fit(X_train,Y_train) # Fit Ridge regression model using reduced number of features

		Y_cap = ridge_object.predict(X_test) # Predict using the linear model

		m = Y_test.shape[0]

		Residual_error += m*mean_squared_error(Y_test,Y_cap) # Mean squared error regression loss

	Residual_error /= 5 # taking average of the 5 errors

	text_file.write("\nFor k : {}, Average Residual Error : {}, No. of features considered : {}".format(k, Residual_error, X_train.shape[1]))
	print("For k : {}, Average Residual Error : {}, No. of features considered : {}".format(k, Residual_error, X_train.shape[1]))
	
	error.append(Residual_error) # Appending the avg error 
	
	k += 0.01 # k incremented by o.5 in each iteration

print("\n\nMinimal error : {} when k = {} \n\n".format(np.min(error),np.argmin(error)*0.01))
text_file.write("\n\nMinimal error : {} when k = {} \n".format(np.min(error),np.argmin(error)*0.01))

text_file.close()

# Minimal error : 1.24915440536 when k = 0.04