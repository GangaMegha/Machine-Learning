############################################################### RIDGE REGRESSION ##################################################################################

import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

Error = [] # List for storing the error values

k = 0 # k stands for lamda values

m = 0 #storing the number of samples.

print("\nRunning code for ridge regression.........\n\n")
print("Output written onto Ridge_lamda.txt and Coefficients.txt\n\n")

text_file = open("Ridge_lamda.txt", "w") # Contains the lamda values used and the corresponding average error obtained
coef_file = open("Coefficients.txt", "w") # Contains the coefficients obtained for each lamda value for each dataset

text_file.write("\nThe file contains the lamda values used and the corresponding average error obtained\n\n")
coef_file.write("\nThis file contains the coefficients obtained for each lamda value for each dataset\n\n")

for j in range(20):	# Iterate over different lamda values. The number of iterations and lamda values have been chosen after a lot of trials 

	k+=0.5 # lamda incremented by 0.5 in each iteration

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
		
		ridge_object = Ridge(alpha=k) # Ridge object initialised with lamda value k

		ridge_object.fit(X_train,Y_train) # Fit Ridge regression model

		Y_cap = ridge_object.predict(X_test) # Predict using the linear model

		m = Y_test.shape[0]

		Residual_error += m*mean_squared_error(Y_test,Y_cap) # Mean squared error regression loss

		coef_file.write("\n\nLamda = {} and Dataset : {}\n".format(k, i))
		coef_file.write("\nCoefficients : {}\n".format(ridge_object.coef_))

	Residual_error /= 5 # Taking average of the errors

	print("Lamda : {}    and    Average Residual Error : {}".format(k, Residual_error))
	text_file.write("\nLamda : {}    and    Average Residual Error : {}".format(k, Residual_error))

	Error.append(Residual_error) # Appending the avg error 

optimal_lamda = np.argmin(Error)*0.5 + 0.5

print("\n\nMinimal Error : {} when Optimal Lamda = {} \n\n".format(np.min(Error), optimal_lamda))
text_file.write("\n\nMinimal error : {} when Optimal Lamda = {} \n\n".format(np.min(Error), optimal_lamda))

text_file.close()
coef_file.close()

#Minimal Error : 1.24972983522 when Optimal Lamda = 5.0 