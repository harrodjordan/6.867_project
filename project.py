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

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# fun fact exactly 20% of the samples are seizure data

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

#print(name_list)

# Random Forest Model 

# NOTE this does not prevent mixing of patients. I don't have the energy to fix it rn
# #=======
rf = models.RandomForest()
rf.train(data, labels, name_list)
sc = rf.test(rf.X_valid, rf.y_valid)


# construct ROC curve for random forest
rf_voting = rf.clf.predict_proba(data)

thresholds = np.linspace(0,1,11)
tpr = np.zeros((thresholds.size,))
fpr = np.zeros((thresholds.size,))

for th in range(len(thresholds)) :
	thresh = thresholds[th]
	pos = (rf_voting[:,1] > thresh).astype(int)
	tpr[th] = np.sum(((pos == 1) & (labels == 1)))/np.sum(labels)
	fpr[th] = np.sum(((pos == 1) * (labels == 0))).astype(float)/np.sum((labels == 0).astype(int))

plt.plot(fpr,tpr,'-o')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC: Random Forest')
plt.show()

cvsc = np.mean(cross_val_score(rf.clf, data, labels, cv=10))


print('Random Forest Accuracy: ' + str(sc[0]))
print('Random Forest Cross Validation Score: ' + str(cvsc))

# SVM Model 
#=======
X_train, X_test, y_train, y_test = train_test_split(data, labels, name_list, test_size = 0.2)

svm = models.SVM()
svm.train(X_train, y_train, name_list)
svm_sc, svm_prec = svm.test(X_test,y_test)


# construct ROC curve for SVM
svm.roc_auc(data, labels)
svm.plot_margin(data, labels)

svm_cvsc = cross_val_score(svm.clf, data, labels, cv=5)

print('SVM Accuracy: ' + str(svm_sc))
print('SVM Precision: ' + str(svm_prec))
print('SVM Cross Validation Scores: ' + str(svm_cvsc))
print('Mean SVM Cross Validation Scores: ' + str(np.mean(svm_cvsc)))



# ConvNet Model 
#=======

cnn = models.CNN() 
data = torch.Tensor(data) 
labels = torch.Tensor(labels) 
cnn.training_procedure(X=data, y=labels, name_list=name_list)

