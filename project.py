import models 
#import keras
import sklearn 
import os
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np

# fun fact exactly 20% of the samples are seizure data

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

rf = models.RandomForest()
rf.train(data, labels)
sc = rf.test(rf.X_valid, rf.y_valid)

cvsc = np.mean(cross_val_score(rf.clf, data, labels, cv=10))

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

svm = models.SVM()
svm.train(data, labels)
svm_sc = svm.test(data,labels)

svm_cvsc = np.mean(cross_val_score(svm.clf, data, labels, cv=10))

print('SVM Accuracy: ' + str(svm_sc))
print('SVM Cross Validation Score: ' + str(svm_cvsc))


