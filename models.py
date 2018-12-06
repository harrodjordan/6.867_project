import sklearn 
import matplotlib.pyplot as plt 
import numpy as np
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score, roc_curve, auc 
# import keras
# from keras.layers import Dense, Activation, Conv1D, Conv2D, MaxPooling1D, Dropout
# from keras.models import Sequential 
# from keras.optimizers import SGD


# class ConvNet():
#     def __init__():
#         self.history = keras.callbacks.History()
#         self.clf = Sequential()
#         self.clf.add(Dense(32, input_dim=178, activation='relu'))
#         self.clf.add(Dropout(0.5))
#         self.clf.add(Dense(64, input_dim=178, activation='relu'))
#         self.clf.add(Dropout(0.5))
#         self.clf.add(Flatten())
#         self.clf.add(Dense(1, input_dim=178, activation='sigmoid'))
#         self.sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#         self.model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=[history])


#     def train(self, X, y, name_list):
#         X_train, X_valid, y_train, y_valid = train_test_split(X, y, name_list, test_size=0.2)
#         self.clf.fit(X_train, y_train, epochs=10, batch_size=115)

#     def test(self, X, y, name_list):
#         X_train, X_valid, y_train, y_valid = train_test_split(X, y, name_list, test_size=0.2)
#         self.clf.evalute(X_valid, y_valid, batch_size=115)


class SVM():

    def __init__(self, C=0.001):
        # initialize classifier
        self.clf = svm.SVC(C=C, class_weight='balanced')


    def train(self, X, y, name_list):
        # fit the classifier
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, name_list, test_size=0.2)
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

    def train(self, X, y, name_list):
        # fit the classifier
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, name_list, test_size=0.2)
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

def train_test_split(X, y, name_list, test_size=0.2) :
    # right now this just splits at the 80% line (no randomness)

    # need to make sure data from a single patient are all in the same category
    # each label is VXX.YYY, but with one to two Xs zero to three Ys
    # so we take XXYYY to be a unique label for a patient (is this true??) (maybe lol)
    # we then sort the data by those labels so that all the data from a given
    # patient is reunited with its source. 

    sort_by = list()

    for x in name_list :
    	labels_split = x.split('.')
    	
    	part1 = labels_split[1][1:]
    	if len(labels_split) == 2 :
    		part2 = ''
    	else :
    		part2 = labels_split[2]

    	sort_by.append(int(part1 + part2))

    order = np.argsort(np.asarray(sort_by))

    X_sorted = X[order]
    y_sorted = y[order]

    # we split at the closest multiple of 23 to the requested split point

    split_at = int(int(int(X.shape[0] * (1 - test_size)) / 23.0) * 23)

    X_train = X[:split_at,:]
    X_valid = X[split_at:,:]
    y_train = y[:split_at]
    y_valid = y[split_at:]
    return X_train, X_valid, y_train, y_valid
