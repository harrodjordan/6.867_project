import models 
import sklearn 
import os
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np


# fun fact exactly 20% of the samples are seizure data

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

# # Random Forest Model 

# rf = models.RandomForest()
# rf.train(data, labels)
# sc, prec = rf.test(data, labels)

# cvsc = np.mean(cross_val_score(rf.clf, data, labels, cv=10))

# print('Random Forest Accuracy: ' + str(sc))
# print('Random Forest Precision: ' + str(prec))
# print('Random Forest Cross Validation Score: ' + str(cvsc))


# SVM Model 

svm = models.SVM()
svm.train(data, labels)
svm_sc, svm_prec = svm.test(data,labels)

svm_cvsc = np.mean(cross_val_score(svm.clf, data, labels, cv=10))

print('SVM Accuracy: ' + str(svm_sc))
print('SVM Precision: ' + str(svm_prec))
print('SVM Cross Validation Score: ' + str(svm_cvsc))

svm.roc_auc(data, labels)

# ConvNet Model (To Come)


