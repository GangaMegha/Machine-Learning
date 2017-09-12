import time
import os
import glob
import tensorflow as tf
import numpy as np
import cv2
import random

# path = 'by_class'
path = 'test'

batch_size=100

t1 = time.time()
file_names=glob.glob(os.path.join(path,'*','train_*','*.[pP][nN][gG]'))
no_of_files=len(file_names)
t2 = time.time()
#print(file_names[0])
print('Time to list files: ', t2-t1)
print('No of files: ',no_of_files)

unique_classes = [int(ele.split('/')[1], base=16) for ele in glob.glob(os.path.join(path,'*/'))]
no_of_classes = len(unique_classes)

labels=[int(ele.split('/')[1], base=16) for ele in file_names]

try:
    label_names = [str(chr(i)) for i in labels]    #python 3
except:
    label_names = [str(unichr(i)) for i in labels]    #python 2.7    

label_encoding = dict()
for idx in range(len(unique_classes)):
    try:
        label_encoding[str(chr(unique_classes[idx]))] = idx
    except:
        label_encoding[str(unichr(unique_classes[idx]))] = idx

print('No of classes: ', no_of_classes)
print('Class encoding: ', label_encoding)



labels_oneHotEncoded = np.zeros((len(file_names),no_of_classes))
for k in range(no_of_files):
	labels_oneHotEncoded[k,label_encoding[label_names[k]]]=1

t3 = time.time()
print('Time to list labels: ', t3-t2)

images = []

for i in range(no_of_files):
    a=np.array(cv2.imread(file_names[i], 0))
    images.append(a.ravel())

images = np.array(images)

t4 = time.time()
print('Time to read images: ',t4-t3)
# Takes about  seconds to read test folder on my 4GB PC :-D
# And the code works!!


x = tf.placeholder(tf.float32, shape=[None, 128*128])
W = tf.Variable(tf.zeros([128*128, no_of_classes]))
b = tf.Variable(tf.zeros([no_of_classes]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, shape=[None, no_of_classes])

print('labels : ',labels_oneHotEncoded)
print('column size : ',images[1].shape)
print('no. of images :', len(images))
cv2.namedWindow('Input',0)
images=images*1.0/255.0
print('non zero :',np.count_nonzero(images[0])) 

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.001,use_locking=False).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#print(correct_prediction)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_list = []
for i in range(1000):
    rand_idx = random.sample(range(no_of_files), batch_size)
    batch_x, batch_y = images[rand_idx], labels_oneHotEncoded[rand_idx]
    
    #Training the NN
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    print('Iteration {:} done'.format(i))
    acc_list.append(sess.run(accuracy, feed_dict={x: images, y_: labels_oneHotEncoded}))

print(max(acc_list))
# print( W[0],x[0])






















import numpy as np 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, f1_score, recall_score 

X = np.genfromtxt('../../Dataset/DS1-train.csv', delimiter=",") # reading train data from the csv file

Y   =   X[:,-2:]    # Y matrix : one hot encoded
X   =   X[:,:-2]    # X matrix

X_test = np.genfromtxt('../../Dataset/DS1-test.csv', delimiter=",") # reading test data from the csv file

np.random.shuffle(X_test)   #shuffling the [X,Y]

Y_test  =   X_test[:,-2:]   # Y matrix : one hot encoded
X_test  =   X_test[:,:-2]   # X matrix

regression_object   =   linear_model.LinearRegression() # Create linear regression object

regression_object.fit(X, Y) # Train the model using the training sets

Y_cap   =   regression_object.predict(X_test) # Predict the output for the test cases

predicted_class = np.argmax(Y_cap, axis=1) # Take the column corresponding to the predicted maximum for each test case

with open("Results.txt", "w") as text_file:
    text_file.write("\nAccuracy : {}".format(accuracy_score(np.array(Y_test[:,1]), np.array(predicted_class))))
    text_file.write("\nPrecision : {}".format(average_precision_score(Y_test[:,1],predicted_class)))
    text_file.write("\nRecall : {}".format(recall_score(Y_test[:,1], predicted_class, average='binary') ))
    text_file.write("\nF-measure : {}".format(f1_score(Y_test[:,1], predicted_class, average='binary') ))

with open("Coefficients.txt", "w") as text_file:
    text_file.write('Coefficients learned : {}'.format(regression_object.coef_))