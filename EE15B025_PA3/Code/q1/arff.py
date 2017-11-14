# For converting data to ARFF format

import numpy as np
import csv

file = ["Aggregation", "Compound", "D31", "Flames", "Jain", "Path-based", "R15", "Spiral"]

for name in file :
	data = np.genfromtxt("../../Dataset/1_Clustering/{}.csv".format(name), delimiter=',')
	print(data[0])
	with open("../../Dataset/{}.arff".format(name), "w") as outfile:
		outfile.write("%Title : {} Dataset".format(name))
		outfile.write("\n@ATTRIBUTE f1 NUMERIC")
		outfile.write("\n@ATTRIBUTE f2 NUMERIC")
		outfile.write("\n\n@ATTRIBUTE 'class' NUMERIC")
		outfile.write("\n\n@DATA")
		for row in data:
			print(row)
			outfile.write("\n")
			for i in range(len(row)-1):
				outfile.write("{},".format(row[i]))
			outfile.write("{}".format(row[2]))
