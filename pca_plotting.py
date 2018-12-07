import models 
import sklearn 
import os
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.decomposition import PCA


data, labels, name_list = feature_extraction.raw_data(two_cat = True)

rf = models.RandomForest()
rf.train(data, labels, name_list)
sc = rf.test(rf.X_valid, rf.y_valid)

labels_predict =  rf.clf.predict(rf.X_valid)

pca = PCA()

data_pca = pca.fit_transform(rf.X_valid)

avges = np.mean(rf.X_valid,axis=1)
sigs = np.std(rf.X_valid,axis=1)

plt.figure()
plt.scatter(avges,sigs,alpha=0.5,c=labels_predict)
plt.xlabel('mean')
plt.ylabel('stdev')
plt.title('Labeling for mean and stdev of each sample (Random Forest)')
plt.show()

plt.figure()
plt.scatter(data_pca[:,0],data_pca[:,1],alpha=0.5,c=labels_predict)
plt.xlabel('pca_1')
plt.ylabel('pca_2')
plt.title('Labeling in principle component space (Random Forest)')
plt.show()

# now, plot the curves
t = np.linspace(0,1000,rf.X_valid.shape[1])
plt.figure(figsize=(12,6))
for i in range(rf.X_valid.shape[0]) :
    if labels_predict[i] :
        plt.plot(t,rf.X_valid[i,:],'-r',alpha=0.3)
    else :
        plt.plot(t,rf.X_valid[i,:],'-b',alpha=0.3)

plt.xlabel('time (ms)')
plt.ylabel('signal')
plt.title('Labeling of signals (Random Forest)')
plt.show()