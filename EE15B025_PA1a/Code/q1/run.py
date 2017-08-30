import numpy as np 

sigma	=	np.random.random((20,20)) 		# covariance matrix
sigma	= 	np.dot(sigma,sigma.transpose())	# positive semidefinite covariance matrix
mu 		=	np.random.random(20) 			# mean vector
dev 	=	np.sqrt(sigma.diagonal())		# taking the square root of the diagonal to be something like the standard deviation
mu_1 	= 	mu + 0.65*dev		# mean for first multivariate gaussian distribution 
mu_2 	= 	mu - 0.65*dev		# mean for second multivariate gaussian distribution 

x1	=	np.vstack((np.ones(2000),np.random.multivariate_normal(mu_1,sigma,2000).transpose())).transpose()	# Draw 2000 random samples from a multivariate normal distribution
x2	=	np.vstack((np.ones(2000),np.random.multivariate_normal(mu_2,sigma,2000).transpose())).transpose()	# Draw 2000 random samples from a multivariate normal distribution

x1 	= 	np.vstack((x1.transpose(),np.vstack((np.ones(2000),np.zeros(2000))))).transpose()	# appending y=[1,0] for samples from first gaussian => [x1,y1] , y1 is one-hot encoded
x2 	= 	np.vstack((x2.transpose(),np.vstack((np.zeros(2000),np.ones(2000))))).transpose()	# appending y=[0,1] for samples from second gaussian => [x2,y2] , y2 is one-hot encoded

np.random.shuffle(x1)	#shuffling the [x1,y1]
np.random.shuffle(x2)	#shuffling the [x2,y2]


train_data_1	=	x1[:1400]	#taking the first 1400 samples for training
test_data_1		=	x1[1400:]	#taking the last 600 samples for testing

train_data_2	=	x2[:1400]	#taking the first 1400 samples for training
test_data_2		=	x2[1400:]	#taking the last 600 samples for testing

X 	= 	np.vstack((train_data_1,train_data_2))	# stacking the two classes together for training => [X,Y]

np.random.shuffle(X)	#shuffling the [X,Y]

np.savetxt("../../Dataset/DS1-train.csv", X, delimiter=',') # Writing the output into a csv file

X_test 	= 	np.vstack((test_data_1,test_data_2))	# stacking the two classes together for testing => [X,Y]

np.random.shuffle(X_test)	#shuffling the [X,Y]

np.savetxt("../../Dataset/DS1-test.csv", X_test, delimiter=',') # Writing the output into a csv file