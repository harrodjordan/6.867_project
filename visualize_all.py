import sklearn 
import os
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.decomposition import PCA


data, labels, name_list = feature_extraction.raw_data(two_cat = True)

X_train, X_test, y_train, y_test = train_test_split(data, labels, name_list, test_size = 0.2)

pca = PCA()

data_pca = pca.fit_transform(data)

avges = np.mean(data,axis=1)
sigs = np.std(data,axis=1)

plt.figure()
plt.scatter(avges,sigs,alpha=0.5,c=labels)
plt.xlabel('mean')
plt.ylabel('stdev')
plt.title('True labels w.r.t. mean and stdev of each sample')
plt.show()

plt.figure()
plt.scatter(data_pca[:,0],data_pca[:,1],alpha=0.5,c=labels)
plt.xlabel('pca_1')
plt.ylabel('pca_2')
plt.title('True labels in principle component space')
plt.show()

# now, plot the curves
t = np.linspace(0,1000,data.shape[1])
plt.figure(figsize=(12,6))
for i in range(data.shape[0]) :
    if labels[i] :
        plt.plot(t,data[i,:],'-r',alpha=0.3)
    else :
        plt.plot(t,data[i,:],'-b',alpha=0.3)

plt.xlabel('time (ms)')
plt.ylabel('signal')
plt.title('True labels of signals')
plt.show()