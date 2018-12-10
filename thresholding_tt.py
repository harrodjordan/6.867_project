from models import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import feature_extraction
from scipy.stats import norm

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

trainX, testX, trainY, testY = train_test_split(data, labels, name_list)

train_sigs = np.std(trainX, axis=1)
test_sigs = np.std(testX, axis=1)


thresh = np.linspace(np.min(train_sigs),np.max(train_sigs),50)
tpr = np.zeros((thresh.shape))
fpr = np.zeros((thresh.shape))

for t in range((thresh).size) :
    th = thresh[t]
    labels_pred = (train_sigs > th)
    tpr[t] = np.sum(((labels_pred == 1) & (trainY == 1)))/np.sum(trainY)
    fpr[t] = np.sum(((labels_pred == 1) * (trainY == 0))).astype(float)/np.sum((trainY == 0).astype(int))

plt.figure()
plt.plot(fpr,tpr,'-o')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig('test.png')


# distance to corner
d = np.sqrt((1-tpr)**2 + (1-(1-fpr))**2)

th_loc = np.argmin(d)

print('The optimal threshold is ' + str(thresh[th_loc]))
print('At this threshold, FPR = ' + str(fpr[th_loc]) + ' and TPR = ' + str(tpr[th_loc]))

# what is the overall accuracy?
pred = (train_sigs > thresh[th_loc])
acc = np.sum(pred == trainY).astype(float)/(trainY.size)
print('Overall, the training accuracy is ' + str(acc))

# how do we do on test data?
pred_test = (test_sigs > thresh[th_loc])
acc_test = np.sum(pred_test == testY).astype(float)/(testY.size)
print('The testing accuracy is ' + str(acc_test))

# plot of all data with line for threshold
sigs = np.std(data, axis=1)
means = np.mean(data, axis=1)

plt.figure()
plt.scatter(means,sigs,c=labels,alpha=0.5)
plt.plot([-200, 300],[thresh[th_loc],thresh[th_loc]],'--g',linewidth=2)
plt.xlim(-200, 300)
plt.ylim(-100,800)
plt.xlabel('mean')
plt.ylabel('standard deviation')
plt.savefig('std_w_thresh.png')
