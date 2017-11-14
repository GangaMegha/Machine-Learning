import numpy as np 
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


with open("Result_nb_ber_beta_sklearn.txt", "w") as text_file:
	f=0
	alpha = float(raw_input("\nEnter value of alpha : "))
	beta = float(raw_input("\nEnter value of beta : "))
	for i in range(5):

		X_train = np.genfromtxt("../../Dataset/2_NaiveBayes/X_train_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)
		X_test = np.genfromtxt("../../Dataset/2_NaiveBayes/X_test_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)
		Y_train = np.genfromtxt("../../Dataset/2_NaiveBayes/Y_train_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)
		Y_test = np.genfromtxt("../../Dataset/2_NaiveBayes/Y_test_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)

		y_train = np.zeros(Y_train.shape[0])
		y_train[np.where(Y_train=="spam")]=1 # Encoding as zeors and ones

		y_test = np.zeros(Y_test.shape[0])
		y_test[np.where(Y_test=="spam")]=1 # Encoding as zeors and ones

		N = X_train.shape[0] #total documents

		prior = np.zeros(2)
		if(Y_train[0]=="legit"):
			Nc = len(X_train[np.where(Y_train=="legit")])
		else:
			Nc = len(X_train[np.where(Y_train=="spam")])

		prior[0] = (Nc+alpha-1)*1.0/(N+alpha+beta-2)
		prior[1] = (N-Nc+beta-1)*1.0/(N+alpha+beta-2)

		clf = BernoulliNB(class_prior=prior)
		clf.fit(X_train, y_train)

		y_prediction = clf.predict(X_test)

		target_names = []
		target_names.append(Y_test[0])
		if Y_test[0]=="legit":
			target_names.append("spam")
		else :
			target_names.append("legit")

		text_file.write("\nCross-Validation {}\n\n".format(i+1))	
		text_file.write(classification_report(y_test, y_prediction, target_names=target_names))
		text_file.write("\n\n\n")
		
		f_new = f1_score(y_test, y_prediction, average='micro') 
		if(f<f_new):
			f = f_new
			y_score = clf.predict_proba(X_test)[:,1]
			y_true = y_test

	# Plot PR Curve
	precision, recall, thresholds = precision_recall_curve(y_true, y_score)
	
	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve')
	plt.show()

print("\n\nDone :)")