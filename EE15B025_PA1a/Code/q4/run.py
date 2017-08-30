import numpy as np 
from sklearn.preprocessing import Imputer #Imputation transformer for completing missing values.

dataset = np.genfromtxt('communities.data.csv', delimiter=",") #reading data from the csv file

dataset_modified = dataset[:,5:] # Taking only the required data (Predictive Attributes)

imputer = Imputer(strategy="mean") # replace missing values using the mean along the axis 0 (column)

transformed_dataset = imputer.fit_transform(dataset_modified) # Fit to data, then transform it

np.savetxt("CandC_modified.csv", transformed_dataset, delimiter=',') # Writing the output into a csv file