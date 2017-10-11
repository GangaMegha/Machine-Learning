# Converting data into libsvm format

import csv

with open("../../Dataset/DS2_train_libsvm", "w") as outfile:
	with open("../../Dataset/DS2_train.csv", "r") as infile:
		reader = csv.reader(infile)
		for row in reader:
			outfile.write("{} ".format(row[len(row)-1]))
			for indx in range(len(row)-1):
				outfile.write("{}:{} ".format(indx+1, row[indx]))
			outfile.write("\n")

with open("../../Dataset/DS2_test_libsvm", "w") as outfile:
	with open("../../Dataset/DS2_test.csv", "r") as infile:
		reader = csv.reader(infile)
		for row in reader:
			outfile.write("{} ".format(row[len(row)-1]))
			for indx in range(len(row)-1):
				outfile.write("{}:{} ".format(indx+1, row[indx]))
			outfile.write("\n")