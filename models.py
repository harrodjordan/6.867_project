import sklearn 
import matplotlib.pyplot as plt 
import numpy as np
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score, roc_curve, auc 


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
        self.clf = svm.SVC(C=C, class_weight='balanced')


    def train(self, X, y):
        # fit the classifier
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.clf = self.clf.fit(X,y)

    def test(self, X, true_labels):
        prediction = self.clf.predict(X)
        # get mean accuracy (not a good metric!)
        score = self.clf.score(X, true_labels)
        precision = precision_score(true_labels, prediction)
        return score, precision 

    def roc_auc(self, X, true):
        pred = self.clf.predict(X)
        fpr, tpr, thresholds = roc_curve(true, pred)
        auc_var = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC (AUC = %0.2f)' % (auc_var))
        plt.show()

    def plot_margin(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        # plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')
        plt.show()



# TODO add to parameter list
class RandomForest():

    def __init__(self, n_estimators=10):
        # initialize classifier
        self.clf = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X, y):
        # fit the classifier
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.clf = self.clf.fit(X_train, y_train)

    # do we want to just spit out labels or testing accuracies?
    # kind of an arbitrary choice
    # we also care about things like TPR etc
    def test(self, X, true_labels):
        # predict labels
        prediction = self.clf.predict(X)
        # get mean accuracy (not a good metric!)
        score = self.clf.score(X, true_labels)
        precision = precision_score(true_labels, prediction)

        return score, precision 

def train_test_split(X, y, test_size=0.2) :
    # right now this just splits at the 80% line (no randomness)
    # need to eventually make sure data from the same individual are in the same group
    split_at = int(X.shape[0] * (1 - test_size))
    X_train = X[:split_at,:]
    X_valid = X[split_at:,:]
    y_train = y[:split_at]
    y_valid = y[split_at:]
    return X_train, X_valid, y_train, y_valid
