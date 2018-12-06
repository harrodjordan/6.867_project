import models 
import sklearn 
import os
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# fun fact exactly 20% of the samples are seizure data

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

#print(name_list)

# # Random Forest Model 

# rf = models.RandomForest()
# rf.train(data, labels)
# sc, prec = rf.test(data, labels)

# NOTE this does not prevent mixing of patients. I don't have the energy to fix it rn
#=======
rf = models.RandomForest()
rf.train(data, labels, name_list)
sc = rf.test(rf.X_valid, rf.y_valid)

# print('Random Forest Accuracy: ' + str(sc))
# print('Random Forest Precision: ' + str(prec))
# print('Random Forest Cross Validation Score: ' + str(cvsc))


# SVM Model 
#=======
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

cvsc = np.mean(cross_val_score(rf.clf, data, labels, cv=10))


print('Random Forest Accuracy: ' + str(sc))
print('Random Forest Cross Validation Score: ' + str(cvsc))

svm = models.SVM()
svm.train(data, labels, name_list)
svm_sc, svm_prec = svm.test(data,labels)

svm_cvsc = np.mean(cross_val_score(svm.clf, data, labels, cv=10))

print('SVM Accuracy: ' + str(svm_sc))
print('SVM Precision: ' + str(svm_prec))
print('SVM Cross Validation Score: ' + str(svm_cvsc))

svm.roc_auc(data, labels)

# ConvNet Model 
cnn = models.ConvNet() 
cnn.train(data, labels, name_list)
cnn.test(data, labels, name_list)

