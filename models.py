import sklearn 
import keras.layers as k
import matplotlib.pyplot as plt 
import numpy 
import os 
from sklearn.ensemble import RandomForestClassifier

class RNN():

	def __init__():
		pass

	def train():
		pass

	def test():
		pass


class SVM():

	def __init__():
		pass

	def train():
		pass

	def test():
		pass


# TODO add to parameter list
class RandomForest(n_estimators=10):

	def __init__():
		# initialize classifier
		self.clf = RandomForestClassifier(n_estimators=n_estimators)

	def train(X,Y):
		# fit the classifier
		self.clf = clf.fit(X,Y)

	# do we want to just spit out labels or testing accuracies?
	# kind of an arbitrary choice
	# we also care about things like TPR etc
	def test(X, true_labels):
		# predict labels
		prediction = self.clf.predict(X)
		# get mean accuracy (not a good metric!)
		score = self.clf.score(X, true_labels)
