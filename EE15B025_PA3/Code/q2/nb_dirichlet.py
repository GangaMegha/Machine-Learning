import numpy as np 
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


with open("Result_nb_dirichlet.txt", "w") as text_file:	
	f=0
	for i in range(5):

		X_train = np.genfromtxt("../../Dataset/2_NaiveBayes/X_train_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)
		X_test = np.genfromtxt("../../Dataset/2_NaiveBayes/X_test_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)
		Y_train = np.genfromtxt("../../Dataset/2_NaiveBayes/Y_train_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)
		Y_test = np.genfromtxt("../../Dataset/2_NaiveBayes/Y_test_{}.csv".format(i+1), dtype=None, delimiter = ",", skip_header=1)

		# Training the Multinomial Naive Bayes
		N = X_train.shape[0] # N = No. of Documents
		V = X_train.shape[1] # vocabulary
		
		labels = {"legit", "spam"}

		prior = np.zeros(len(labels))
		condprob = np.zeros((V, len(labels)))

		for j,label in zip(range(len(labels)), labels):
			Nc = len(np.where(Y_train==label)[0])
			prior[j] = 1.0*Nc/N
			textc = np.where(np.sum(X_train[np.where(Y_train==label)], axis=0)>0) # Words in class label
			Tct = np.zeros(V)
			for g in range(V):
				if g in textc[0]:
					Tct[g] = np.sum(X_train, axis=0)[g] #  Count of each word
			
			alpha = np.multiply(np.random.random(len(Tct)), Tct) # Multiplying each word count with a random fraction

			condprob[:,j] = ( Tct + alpha + np.ones_like(Tct) ) *1.0/ ( sum(Tct) + sum(alpha) - len(alpha) ) # Conditional Probabilities


		# Apply Multinomial Naive Bayes 
		W = X_test.shape[1]
		score = np.zeros((X_test.shape[0], 2))
		score[:] = np.log(prior)
		for k in range(X_test.shape[0]):
			for m in range(2):
				temp = np.multiply(X_test[k].reshape(condprob.shape[0]), np.log(condprob[:,m]))
				score[k][m] = score[k][m] + np.sum(temp[np.where(X_test[k]>0)], axis=0)
		
		y_prediction = np.argmax(score, axis=1)	# Output prediction
		target_names = []
		target_names.append(Y_test[0])
		if Y_test[0]=="legit":
			target_names.append("spam")
		else :
			target_names.append("legit")

		y_true = np.zeros(Y_test.shape[0])
		y_true[np.where(Y_test=="spam")]=1 # Encoding as zeors and ones

		text_file.write("\nCross-Validation {}\n\n".format(i+1))	
		text_file.write(classification_report(y_true, y_prediction, target_names=target_names))
		text_file.write("\n\n\n")	

		f_new = f1_score(y_true, y_prediction, average='micro') 
		if(f<f_new):
			f = f_new
			# y_score = (np.amin(score)*(-1)*np.ones(score.shape[0]) + score[:,1])/(np.amin(score)*(-1))
			y_score = np.divide(score[:,0],score[:,0]+score[:,1])
			y_test = y_true


	# Plot PR Curve
	precision, recall, thresholds = precision_recall_curve(y_test, y_score)
	
	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve')
	plt.show()

print("\n\nDone :)")