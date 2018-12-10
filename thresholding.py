from models import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import feature_extraction
from scipy.stats import norm

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

sigs = np.std(data,axis=1)


thresh = np.linspace(np.min(sigs),np.max(sigs),50)
tpr = np.zeros((thresh.shape))
fpr = np.zeros((thresh.shape))

for t in range((thresh).size) :
    th = thresh[t]
    labels_pred = (sigs > th)
    tpr[t] = np.sum(((labels_pred == 1) & (labels == 1)))/np.sum(labels)
    fpr[t] = np.sum(((labels_pred == 1) * (labels == 0))).astype(float)/np.sum((labels == 0).astype(int))

plt.figure()
plt.plot(fpr,tpr,'-o')
plt.savefig('test.png')

# distance to corner
d = np.sqrt((1-tpr)**2 + (1-(1-fpr))**2)

th_loc = np.argmin(d)

print('The optimal threshold is ' + str(thresh[th_loc]))
print('At this threshold, FPR = ' + str(fpr[th_loc]) + ' and TPR = ' + str(tpr[th_loc]))

# what is the overall accuracy?
pred = (sigs > thresh[th_loc])
acc = np.sum(pred == labels).astype(float)/(labels.size)
print('Overall, the accuracy is ' + str(acc))