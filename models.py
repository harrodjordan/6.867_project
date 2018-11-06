import sklearn 
import keras.layers as k
import matplotlib.pyplot as plt 
import numpy 
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pywt


class ConvNet():

	def __init__():
		pass

	def train():
		pass

	def test():
		pass


class SVM():

	def __init__(self, C=0.001):
		# initialize classifier
		self.clf = svm.LinearSVC(C=C)
		

	def train():
		# fit the classifier
		X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
		self.clf = self.clf.fit(X,Y)

	def test():
		prediction = self.clf.predict(X)
		# get mean accuracy (not a good metric!)
		score = self.clf.score(X, true_labels)



# TODO add to parameter list
class RandomForest():

	def __init__(self, n_estimators=10):
		# initialize classifier
		self.clf = RandomForestClassifier(n_estimators=n_estimators)

	def train(self, X, y):
		# fit the classifier
		X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
		self.clf = self.clf.fit(X_train, y_train)

	# do we want to just spit out labels or testing accuracies?
	# kind of an arbitrary choice
	# we also care about things like TPR etc
	def test(self, X, true_labels):
		# predict labels
		prediction = self.clf.predict(X)
		# get mean accuracy (not a good metric!)
		score = self.clf.score(X, true_labels)

		return score

def train_test_split(X, y, test_size=0.2) :
	# right now this just splits at the 80% line (no randomness)
	# need to eventually make sure data from the same individual are in the same group
	split_at = int(X.shape[0] * (1 - test_size))
	X_train = X[:split_at,:]
	X_valid = X[split_at:,:]
	y_train = y[:split_at]
	y_valid = y[split_at:]
	return X_train, X_valid, y_train, y_valid
