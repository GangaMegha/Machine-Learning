import numpy as np 
import glob
import os
import csv

from collections import Counter
from string import punctuation

path = "../../Dataset/2_NaiveBayes/"  

k = 0
for i in {1,3,5,7,9} :

	k = k+1 #Cross-validation set number

	# Storing all the text file paths for train and test seperately
	filenames_test = glob.glob(os.path.join(path,"*{}".format(i),"*.txt"))
	np.hstack((filenames_test, glob.glob(os.path.join(path,"*{}".format(i+1),"*.txt"))))
	for j in range(10):
		if(j!=i and j!=(i+1)):
			filenames_train = glob.glob(os.path.join(path,"*{}".format(j),"*.txt"))
	
	# Making an array of dictionaries for counting frequency of occurance of each word 
	train = list( {} for j in xrange(len(filenames_train)) )
	test = list( {} for j in xrange(len(filenames_test)) )

	train_labels = []
	test_labels = []

	# Making the Train Matrix
	for num,file in zip(np.arange(len(filenames_train)), filenames_train):
		with open(file) as f:
			# Adding the new word to the dictionary and incrementing counter
			for line in f:
				for word in line.lower().split():
					key = word.rstrip(punctuation)
					if(key != "subject"):
						if key in train[num]:
							train[num][key] = train[num][key] + 1
						else:
							for m in range(len(train)):
								train[m][key] = 0
						# if key not in test[0]:
							for m in range(len(test)):
								test[m][key] = 0

			# Adding corresponding labels 
			if("legit" in file.split('/')[5].split('.')[0]):
				train_labels.append("legit")
			else:
				train_labels.append("spam")

	# Making the Test Matrix
	for num,file in zip(np.arange(len(filenames_test)), filenames_test):
		with open(file) as f:

			# Adding the new word to the dictionary and incrementing counter
			for line in f:
				for word in line.lower().split():
					key = word.rstrip(punctuation)
					if(key != "subject"):
						if key in test[num]:
							test[num][key] = test[num][key] + 1

			# Adding corresponding labels 
			if("legit" in file.split('/')[5].split('.')[0]):
				test_labels.append("legit")
			else:
				test_labels.append("spam")
	
	train_keys = train[0].keys()
	test_keys = test[0].keys()
	print("\n\n")
	print(len(train_keys))
	print(len(test_keys))

	with open('../../Dataset/2_NaiveBayes/X_train_{}.csv'.format(k), 'wb') as output_file:
		dict_writer = csv.DictWriter(output_file, train_keys)
		dict_writer.writeheader()
		dict_writer.writerows(train)

	with open('../../Dataset/2_NaiveBayes/X_test_{}.csv'.format(k), 'wb') as output_file:
		dict_writer = csv.DictWriter(output_file, test_keys)
		dict_writer.writeheader()
		dict_writer.writerows(test)

	np.savetxt("../../Dataset/2_NaiveBayes/Y_train_{}.csv".format(k), np.array(train_labels),  fmt='%s', delimiter=',', newline='\n', header='label',comments='')
	np.savetxt("../../Dataset/2_NaiveBayes/Y_test_{}.csv".format(k), np.array(test_labels),  fmt='%s', delimiter=',', newline='\n', header='label',comments='')

