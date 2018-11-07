import models 
import sklearn 
import os
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np


# fun fact exactly 20% of the samples are seizure data

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

<<<<<<< HEAD
# # Random Forest Model 

# rf = models.RandomForest()
# rf.train(data, labels)
# sc, prec = rf.test(data, labels)

# cvsc = np.mean(cross_val_score(rf.clf, data, labels, cv=10))
=======
rf = models.RandomForest()
rf.train(data, labels)
sc = rf.test(rf.X_valid, rf.y_valid)
>>>>>>> 54a28e808d8e173343499aa4c750bc15ea381287

# print('Random Forest Accuracy: ' + str(sc))
# print('Random Forest Precision: ' + str(prec))
# print('Random Forest Cross Validation Score: ' + str(cvsc))

<<<<<<< HEAD

# SVM Model 
=======
# construct ROC curve for random forest
rf_voting = rf.clf.predict_proba(data)

thresholds = np.linspace(0,1,11)
tpr = np.zeros((thresholds.size,))
fpr = np.zeros((thresholds.size,))

for th in range(len(thresholds)) :
	thresh = thresholds[th]
	pos = (rf_voting[:,1] > thresh).astype(int)
	tpr[th] = np.sum(((pos == 1) & (labels == 1)))/np.sum(labels)
	fpr[th] = np.sum(((pos == 1) * (labels == 0)))/np.sum((labels == 0).astype(int))

plt.plot(fpr,tpr,'-o')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC: Random Forest')
plt.show()

print('Random Forest Accuracy: ' + str(sc))
print('Random Forest Cross Validation Score: ' + str(cvsc))
>>>>>>> 54a28e808d8e173343499aa4c750bc15ea381287

svm = models.SVM()
svm.train(data, labels)
svm_sc, svm_prec = svm.test(data,labels)

svm_cvsc = np.mean(cross_val_score(svm.clf, data, labels, cv=10))

print('SVM Accuracy: ' + str(svm_sc))
print('SVM Precision: ' + str(svm_prec))
print('SVM Cross Validation Score: ' + str(svm_cvsc))

svm.roc_auc(data, labels)

# ConvNet Model (To Come)


