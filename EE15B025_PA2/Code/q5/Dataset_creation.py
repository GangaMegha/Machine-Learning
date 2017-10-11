# For converting data to ARFF format

import numpy as np
import csv

data = np.array(list(csv.reader(open("../../Dataset/agaricus-lepiota.data.csv", "rb"), delimiter=",")))
train_data = data[:7000]
test_data = data[7000:]

with open("../../Dataset/mushroom_train.arff", "w") as outfile:
	outfile.write("%Title : Mushroom Dataset")
	outfile.write("\n\n@RELATION mushroom")
	outfile.write("\n@ATTRIBUTE 'cap-shape' {'b', 'c', 'x', 'f', 'k', 's'}")
	outfile.write("\n@ATTRIBUTE 'cap-surface' {'f', 'g', 'y', 's'}")
	outfile.write("\n@ATTRIBUTE 'cap-color' {'n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'bruises?' {'t', 'f'}")
	outfile.write("\n@ATTRIBUTE 'odor' {'a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'}")
	outfile.write("\n@ATTRIBUTE 'grill-attachment' {'a', 'd', 'f', 'n'}")
	outfile.write("\n@ATTRIBUTE 'grill-spacing' {'c', 'w', 'd'}")
	outfile.write("\n@ATTRIBUTE 'grill-size' {'b', 'n'}")
	outfile.write("\n@ATTRIBUTE 'grill-color' {'k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'stalk-shape' {'e', 't'}")
	outfile.write("\n@ATTRIBUTE 'stalk-root' {'b', 'c', 'u', 'e', 'z', 'r'}")
	outfile.write("\n@ATTRIBUTE 'stalk-surface-above-ring' {'f', 'y', 'k', 's'}")
	outfile.write("\n@ATTRIBUTE 'stalk-surface-below-ring' {'f', 'y', 'k', 's'}")
	outfile.write("\n@ATTRIBUTE 'stalk-color-above-ring' {'n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'stalk-color-below-ring' {'n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'veil-type' {'p', 'u'}")
	outfile.write("\n@ATTRIBUTE 'veil-color' {'n', 'o', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'ring-number' {'n', 'o', 't'}")
	outfile.write("\n@ATTRIBUTE 'ring-type' {'c', 'e', 'f', 'l', 'n', 'p', 's', 'z'}")
	outfile.write("\n@ATTRIBUTE 'spore-print-color' {'k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'population' {'a', 'c', 'n', 's', 'v', 'y'}")
	outfile.write("\n@ATTRIBUTE 'habitat' {'g', 'l', 'm', 'p', 'u', 'w', 'd'}")
	outfile.write("\n\n@ATTRIBUTE 'class' {'e', 'p'}")
	outfile.write("\n\n@DATA")
	for row in train_data:
		outfile.write("\n")
		for i, item in zip(range(len(row)-1), row[1:]):
			if item!='?':
				outfile.write("'{}',".format(item))
			else:
				outfile.write("{}".format(item))
		outfile.write("'{}'".format(row[0]))

with open("../../Dataset/mushroom_test.arff", "w") as outfile:
	outfile.write("%Title : Mushroom Dataset")
	outfile.write("\n\n@RELATION mushroom")
	outfile.write("\n@ATTRIBUTE 'cap-shape' {'b', 'c', 'x', 'f', 'k', 's'}")
	outfile.write("\n@ATTRIBUTE 'cap-surface' {'f', 'g', 'y', 's'}")
	outfile.write("\n@ATTRIBUTE 'cap-color' {'n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'bruises?' {'t', 'f'}")
	outfile.write("\n@ATTRIBUTE 'odor' {'a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'}")
	outfile.write("\n@ATTRIBUTE 'grill-attachment' {'a', 'd', 'f', 'n'}")
	outfile.write("\n@ATTRIBUTE 'grill-spacing' {'c', 'w', 'd'}")
	outfile.write("\n@ATTRIBUTE 'grill-size' {'b', 'n'}")
	outfile.write("\n@ATTRIBUTE 'grill-color' {'k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'stalk-shape' {'e', 't'}")
	outfile.write("\n@ATTRIBUTE 'stalk-root' {'b', 'c', 'u', 'e', 'z', 'r'}")
	outfile.write("\n@ATTRIBUTE 'stalk-surface-above-ring' {'f', 'y', 'k', 's'}")
	outfile.write("\n@ATTRIBUTE 'stalk-surface-below-ring' {'f', 'y', 'k', 's'}")
	outfile.write("\n@ATTRIBUTE 'stalk-color-above-ring' {'n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'stalk-color-below-ring' {'n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'veil-type' {'p', 'u'}")
	outfile.write("\n@ATTRIBUTE 'veil-color' {'n', 'o', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'ring-number' {'n', 'o', 't'}")
	outfile.write("\n@ATTRIBUTE 'ring-type' {'c', 'e', 'f', 'l', 'n', 'p', 's', 'z'}")
	outfile.write("\n@ATTRIBUTE 'spore-print-color' {'k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'}")
	outfile.write("\n@ATTRIBUTE 'population' {'a', 'c', 'n', 's', 'v', 'y'}")
	outfile.write("\n@ATTRIBUTE 'habitat' {'g', 'l', 'm', 'p', 'u', 'w', 'd'}")
	outfile.write("\n\n@ATTRIBUTE 'class' {'e', 'p'}")
	outfile.write("\n\n@DATA")
	for row in test_data:
		outfile.write("\n")
		for i, item in zip(range(len(row)-1), row[1:]):
			if item!='?':
				outfile.write("'{}',".format(item))
			else:
				outfile.write("{}".format(item))
		outfile.write("'{}'".format(row[0]))


