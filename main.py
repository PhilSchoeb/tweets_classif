# IFT 3335 - TP 2
# Code de Philippe Schoeb et Nathan Bussière
# 19 avril 2024

import numpy as np
from sklearn.model_selection import train_test_split as split
import preprocess
from preprocess import DataReader
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import f1_score, accuracy_score
from sklearn import tree, ensemble, svm, neural_network
import sys
import time

# Get data
# Gotta put the directory to access data file (only training because the test_data file doesnt have labels)
reader = DataReader('./datasets/offenseval-training-v1.tsv')
data, labels = reader.get_labelled_data()
data, labels = reader.shuffle(data, labels, 'random')
print(np.shape(data))
print(np.shape(labels))
data = preprocess.preprocess_data(data)
print(np.shape(data))
tr_data, tst_data, tr_labels, tst_labels = split(data, labels, test_size=0.3, random_state=16)
tr_data = tr_data.toarray()
tst_data = tst_data.toarray()

# Machine learning models

# NAIVE BAYES ##########################################################################################################

gnb = GaussianNB()
t1 = time.time()
pred = gnb.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
GNB_acc = accuracy_score(tst_labels, pred)
GNB_f1 = f1_score(tst_labels, pred)
print('Gaussian Naive Bayes : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' % (t2 - t1, GNB_acc, GNB_f1))

mnb = MultinomialNB()
t1 = time.time()
pred = mnb.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
MNB_acc = accuracy_score(tst_labels, pred)
MNB_f1 = f1_score(tst_labels, pred)
print('Multinomial Naive Bayes : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' % (t2 - t1, MNB_acc, MNB_f1))

# DECISION TREE ########################################################################################################

# Gotta add max_features because 14204 features is too much for training
dtc = tree.DecisionTreeClassifier(criterion='gini', max_features=1000, random_state=16)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (gini, no depth limit, max_features=1000) : Time = %.2f, Accuracy = %.2f, and F1 = '
'%.2f' % (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='entropy', max_features=1000, random_state=16)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (entropy, no depth limit, max_features-1000) : Time = %.2f, Accuracy = %.2f, and F1 = '
'%.2f' % (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='log_loss', max_features=1000, random_state=16)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (log_loss, no depth limit, max_features=1000) : Time = %.2f, Accuracy = %.2f, and F1 = '
      '%.2f' % (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=1)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (gini, max_depth=1) : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' %
      (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (entropy, max_depth=1) : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' %
      (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='log_loss', max_depth=1)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (log_loss, max_depth=1) : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' %
      (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (entropy, max_depth=2) : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' %
      (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (entropy, max_depth=5) : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' %
      (t2 - t1, DTC_acc, DTC_f1))

dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
t1 = time.time()
pred = dtc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
DTC_acc = accuracy_score(tst_labels, pred)
DTC_f1 = f1_score(tst_labels, pred)
print('Decision Tree Classifier (entropy, max_depth=10) : Time = %.2f, Accuracy = %.2f, and F1 = %.2f' %
      (t2 - t1, DTC_acc, DTC_f1))

# RANDOM FOREST ########################################################################################################

# Can't really go with more than 100 trees, training is too long, even with max_features=3
rfc = ensemble.RandomForestClassifier(n_estimators=100, max_features=10, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (100 trees, max_features=10) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' % 
      (t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=10, max_features=200, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (10 trees, max_features=100) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
      (t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=5, max_features=500, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (5 trees, max_features=500) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' % 
(t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=200, max_depth=1, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (200 trees, max_depth=1) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=100, max_depth=1, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (100 trees, max_depth=1) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

# Any max_depth under 10 gives the same accuracy and f1_score = 0. ---> Majority vote
rfc = ensemble.RandomForestClassifier(n_estimators=100, max_depth=15, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (100 trees, max_depth=15) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

# It seems these models have difficulties detecting one class in particular
rfc = ensemble.RandomForestClassifier(n_estimators=100, max_depth=20, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (100 trees, max_depth=20) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=50, max_depth=20, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (50 trees, max_depth=20) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=10, max_depth=20, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (10 trees, max_depth=20) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=10, max_depth=75, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (10 trees, max_depth=75) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=5, max_depth=125, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (5 trees, max_depth=100) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

rfc = ensemble.RandomForestClassifier(n_estimators=3, max_depth=150, random_state=16)
t1 = time.time()
pred = rfc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
RFC_acc = accuracy_score(tst_labels, pred)
RFC_f1 = f1_score(tst_labels, pred)
print('Random Forest Classifier (3 trees, max_depth=150) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, RFC_acc, RFC_f1))

# SVM ##################################################################################################################

svmc = svm.SVC(kernel='linear', max_iter=100, random_state=16)
t1 = time.time()
pred = svmc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SVM_acc = accuracy_score(tst_labels, pred)
SVM_f1 = f1_score(tst_labels, pred)
print('SVM (linear kernel, max_iter=100) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SVM_acc, SVM_f1))

svmc = svm.SVC(kernel='poly', max_iter=100, random_state=16)
t1 = time.time()
pred = svmc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SVM_acc = accuracy_score(tst_labels, pred)
SVM_f1 = f1_score(tst_labels, pred)
print('SVM (poly kernel, max_iter=100) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SVM_acc, SVM_f1))

svmc = svm.SVC(kernel='rbf', max_iter=100, random_state=16)
t1 = time.time()
pred = svmc.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SVM_acc = accuracy_score(tst_labels, pred)
SVM_f1 = f1_score(tst_labels, pred)
print('SVM (rbf kernel, max_iter=100) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SVM_acc, SVM_f1))

# SINGLE LAYER PERCEPTRON ##############################################################################################

slp = neural_network.MLPClassifier(hidden_layer_sizes=[1], activation='relu', random_state=16)
t1 = time.time()
pred = slp.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SLP_acc = accuracy_score(tst_labels, pred)
SLP_f1 = f1_score(tst_labels, pred)
print('Single Layer Perceptron (hidden_layer_size=1, relu) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SLP_acc, SLP_f1))

slp = neural_network.MLPClassifier(hidden_layer_sizes=[2], activation='relu', random_state=16)
t1 = time.time()
pred = slp.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SLP_acc = accuracy_score(tst_labels, pred)
SLP_f1 = f1_score(tst_labels, pred)
print('Single Layer Perceptron (hidden_layer_size=2, relu) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SLP_acc, SLP_f1))

slp = neural_network.MLPClassifier(hidden_layer_sizes=[5], activation='relu', random_state=16)
t1 = time.time()
pred = slp.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SLP_acc = accuracy_score(tst_labels, pred)
SLP_f1 = f1_score(tst_labels, pred)
print('Single Layer Perceptron (hidden_layer_size=5, relu) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SLP_acc, SLP_f1))

# Same results as 5 neurons in hidden layer
slp = neural_network.MLPClassifier(hidden_layer_sizes=[10], activation='relu', random_state=16)
t1 = time.time()
pred = slp.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SLP_acc = accuracy_score(tst_labels, pred)
SLP_f1 = f1_score(tst_labels, pred)
print('Single Layer Perceptron (hidden_layer_size=10, relu) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SLP_acc, SLP_f1))

slp = neural_network.MLPClassifier(hidden_layer_sizes=[2], activation='logistic', random_state=16)
t1 = time.time()
pred = slp.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SLP_acc = accuracy_score(tst_labels, pred)
SLP_f1 = f1_score(tst_labels, pred)
print('Single Layer Perceptron (hidden_layer_size=2, logistic) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SLP_acc, SLP_f1))

slp = neural_network.MLPClassifier(hidden_layer_sizes=[5], activation='logistic', random_state=16)
t1 = time.time()
pred = slp.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SLP_acc = accuracy_score(tst_labels, pred)
SLP_f1 = f1_score(tst_labels, pred)
print('Single Layer Perceptron (hidden_layer_size=5, logistic) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SLP_acc, SLP_f1))

slp = neural_network.MLPClassifier(hidden_layer_sizes=[10], activation='logistic', random_state=16)
t1 = time.time()
pred = slp.fit(tr_data, tr_labels).predict(tst_data)
t2 = time.time()
SLP_acc = accuracy_score(tst_labels, pred)
SLP_f1 = f1_score(tst_labels, pred)
print('Single Layer Perceptron (hidden_layer_size=10, logistic) : Time = %.2f seconds, Accuracy = %.2f, and F1 = %.2f' %
(t2 - t1, SLP_acc, SLP_f1))