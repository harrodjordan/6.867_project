import models 
import sklearn 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np
import torch
import models
from models import train_test_split
from sklearn.metrics import precision_score

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

tree_counts = [5, 10, 25, 50, 100, 1000]

plt.figure()

for t in tree_counts :
    rf = models.RandomForest(n_estimators=t)
    rf.train(data, labels, name_list)
    sc = rf.test(rf.X_valid, rf.y_valid)


    # construct ROC curve for random forest
    rf_voting = rf.clf.predict_proba(data)

    thresholds = np.linspace(0,1,t+1)
    tpr = np.zeros((thresholds.size,))
    fpr = np.zeros((thresholds.size,))

    for th in range(len(thresholds)) :
        thresh = thresholds[th]
        pos = (rf_voting[:,1] > thresh).astype(int)
        tpr[th] = np.sum(((pos == 1) & (labels == 1)))/np.sum(labels)
        fpr[th] = np.sum(((pos == 1) * (labels == 0))).astype(float)/np.sum((labels == 0).astype(int))

    plt.plot(fpr,tpr,'-o')

    cvsc = np.mean(cross_val_score(rf.clf, data, labels, cv=10))

    test_labels = rf.clf.predict(rf.X_valid)

    print('Random Forest Accuracy: ' + str(sc[0]) + ' (' + str(t) + ' trees)')
    print('Random Forest Precision: ' + str(precision_score(rf.y_valid, test_labels)) + ' (' + str(t) + ' trees)')
    print('Random Forest Cross Validation Score: ' + str(cvsc) + ' (' + str(t) + ' trees)')

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC: Random Forest')
plt.legend(tree_counts)
plt.savefig('rf_varying_trees.png')