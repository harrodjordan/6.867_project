# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import scipy.io as sio
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm

# Any results you write to the current directory are saved as output.

# Extracting feature vectors from data
# file structure - input/patient_#/ictal train or non-ictal train or test 

ictal_training_variance = []
nonictal_training_variance= []
test_variance = []

ictal_training_linelength = []
nonictal_training_linelength = []
test_linelength = []

ictal_training_energy = []
nonictal_training_energy = []
test_energy = []

ictal_training_hfo = []
nonictal_training_hfo= []
test_hfo = []

ictal_training_beta = []
nonictal_training_beta = []
test_beta = []

test_names = []

file_path1 = "/Users/jordanharrod/Desktop/data/patient_"

def getint(name):
 
    num = name.split('_')
    num = num[2].split('.')
    
    try:
        return int(num[0])
    except ValueError:
        return -1
        
def getint3(name):
 
    num = name.split('_')


    try: 
        num = num[3].split('.')
    except IndexError:
        return -1
    
    try:
        return int(num[0])
    except ValueError:
        return -1



for x in range(7):

    
    nonictal_files = os.listdir((str(file_path1) + str((x+1)) + "/non-ictal train/"))
    nonictal_files.sort(key=getint) 
    
    
    ictal_files = os.listdir((str(file_path1) + str((x+1)) + "/ictal train/"))
    ictal_files.sort(key=getint)
    
    test_files = os.listdir((str(file_path1) + str((x+1)) + "/test/"))
    test_files.sort(key=getint3)
    
    test_names = test_names + test_files 
    
    print("Loading files from patient " + str(x+1))
    
    for i in range(len(nonictal_files)):
        

        
        file_name = str(nonictal_files[i])
        first = file_name.split("_")
        

        if first[0] != 'patient':
            
            continue 
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/non-ictal train/" + str(nonictal_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])
        
        #print(temp.shape)
        
        length = len(temp)
        
        
        #variance
        
        variance = np.sum(np.var(temp))
        #print(variance.shape)
        nonictal_training_variance = np.append(nonictal_training_variance, variance)
        
        #energy
        
        energy = np.sum(np.square(temp))
        #print(energy.shape)
        nonictal_training_energy = np.append(nonictal_training_energy, energy)
        
        #line length
        
        L = 0
        
        for j in range(length-1):
            L = L + (temp[j+1,:] - temp[j,:]);

        line_length = np.sum(L/length);
        
        #print(line_length.shape)
        nonictal_training_linelength = np.append(nonictal_training_linelength, line_length)
        
        #power spectrum density


        N = len(temp);
        half = round(length/2)
        Fs = length
        xdft = np.fft.fft(temp);
        xdft = xdft[1:half];
        psdx = (1/(Fs*N)) * np.abs(xdft);
        psdx[2:-1] = 2*psdx[2:-1];
        psdx = 10*np.log10(psdx);
        
        beta = np.sum(psdx[12:30]);
        #print(beta.shape)
        nonictal_training_beta = np.append(nonictal_training_beta, beta)
        
        hfo = np.sum(psdx[100:600]);
        #print(hfo.shape)
        nonictal_training_hfo = np.append(nonictal_training_hfo, hfo)
        
    for i in range(len(ictal_files) ):
        
        
        file_name = str(ictal_files[i])
        first = file_name.split("_")
        
        
        
        if first[0] != "patient":
            
            continue 
    
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/ictal train/" + str(ictal_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])
        
        length = len(temp)
        
        #variance
        
        variance = np.sum(np.var(temp, axis=0))
        ictal_training_variance = np.append(ictal_training_variance,variance)
        
        #energy
        
        energy = np.sum(np.square(temp))
        ictal_training_energy = np.append(ictal_training_energy,energy)
        
        #line length
        
        L = 0
        
        for j in range(length-1):
            L = L + (temp[j+1,:] - temp[j,:]);

        line_length = np.sum(L/length);
        ictal_training_linelength = np.append(ictal_training_linelength,line_length)
        
        #power spectrum density


        N = len(temp);
        Fs = length
        half = round(length/2)
        xdft = np.fft.fft(temp);
        xdft = xdft[1:half];
        psdx = (1/(Fs*N)) * np.abs(xdft);
        psdx[2:-1] = 2*psdx[2:-1];
        psdx = 10*np.log10(psdx);
        
        beta = np.sum(psdx[12:30]);
        ictal_training_beta = np.append(ictal_training_beta, beta)
        
        hfo = np.sum(psdx[100:600]);
        ictal_training_hfo = np.append(ictal_training_hfo, hfo)
        
    for i in range(len(test_files)):
        
        file_name = str(test_files[i])
        first = file_name.split("_")
        
        if first[0] != "patient":
            
            continue 
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/test/" + str(test_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])
        
        length = len(temp)
        
        #variance
        
        variance = np.sum(np.var(temp, axis=0))
        test_variance = np.append(test_variance,variance)
        
        #energy
        
        energy = np.sum(np.square(temp))
        test_energy = np.append(test_energy,energy)
        
        #line length
        
        L = 0
        
        for j in range(length-1):
            L = L + (temp[j+1,:] - temp[j,:]);

        line_length = np.sum(L/length);
        test_linelength = np.append(test_linelength,line_length)
        
        #power spectrum density

        N = len(temp);
        Fs = length
        half = round(length/2)
        xdft = np.fft.fft(temp);
        xdft = xdft[1:half];
        psdx = (1/(Fs*N)) * np.abs(xdft);
        psdx[2:-1] = 2*psdx[2:-1];
        psdx = 10*np.log10(psdx);
        
        beta = np.sum(psdx[12:30]);
        test_beta = np.append(test_beta, beta)
        
        hfo = np.sum(psdx[100:600]);
        test_hfo = np.append(test_hfo, hfo)

        
        
clf = svm.LinearSVC(class_weight='balanced')

ictal_training_variance = np.asarray(ictal_training_variance)
nonictal_training_variance= np.asarray(nonictal_training_variance)
test_variance = np.asarray(test_variance)

ictal_training_linelength = np.asarray(ictal_training_linelength)
nonictal_training_linelength = np.asarray(nonictal_training_linelength)
test_linelength = np.asarray(test_linelength)

ictal_training_energy = np.asarray(ictal_training_energy)
nonictal_training_energy = np.asarray(nonictal_training_energy)
test_energy = np.asarray(test_energy)

ictal_training_hfo = np.asarray(ictal_training_hfo)
nonictal_training_hfo= np.asarray(nonictal_training_hfo)
test_hfo = np.asarray(test_hfo)

ictal_training_beta = np.asarray(ictal_training_beta)
nonictal_training_beta = np.asarray(nonictal_training_beta)
test_beta = np.asarray(test_beta)

variance_training = np.append(np.repeat(ictal_training_variance,5), nonictal_training_variance)
print(variance_training.shape)

energy_training = np.append(np.repeat(ictal_training_energy,5),nonictal_training_energy)
print(energy_training.shape)

hfo_training = np.append(np.repeat(ictal_training_hfo,5),nonictal_training_hfo)
print(hfo_training.shape)

beta_training = np.append(np.repeat(ictal_training_beta,5),nonictal_training_beta)
print(beta_training.shape)

line_training = np.append(np.repeat(ictal_training_linelength,5), nonictal_training_linelength)
print(line_training.shape)

X_training = np.column_stack((variance_training, energy_training, line_training, hfo_training, beta_training))

X_training = np.asarray(X_training, dtype=np.float64)

where_are_nans = np.isnan(X_training)
X_training[where_are_nans] = 0
where_are_inf = np.isinf(X_training)
X_training[where_are_inf] = 100




X_training = np.asarray(X_training, dtype=np.float64)

print(X_training.shape)

nonictal_training = np.column_stack((nonictal_training_variance, nonictal_training_energy, nonictal_training_linelength, nonictal_training_hfo, nonictal_training_beta))

test = np.column_stack((test_variance, test_energy, test_linelength, test_hfo, test_beta))

test = np.asarray(test)

where_are_nans = np.isnan(test)
test[where_are_nans] = 0
where_are_inf = np.isinf(test)
test[where_are_inf] = 100

test = np.asarray(test)

#X = np.stack((ictal_training, nonictal_training))

ictal_labels = np.ones(len(np.repeat(ictal_training_variance,5)))
nonictal_labels = np.zeros(len(nonictal_training_variance))

y = np.append(ictal_labels, nonictal_labels)
y = np.asarray(y)
print(y.shape)

X_training = np.asarray(X_training)

where_are_nans = np.isnan(X_training)
X_training[where_are_nans] = 0
where_are_inf = np.isinf(X_training)
X_training[where_are_inf] = 100




print(X_training.shape)



X_train, X_valid, y_train, y_valid = train_test_split(X_training, y, test_size=0.4)

print("Fitting the SVM")

clf.fit(X_train, y_train)  

print("Model Complete. Predicting from Test Data")

print(clf.score(X_valid, y_valid))


print("Model Complete. Predicting from Test Data")
test_results = clf.predict(test)

for result in range(len(test_results)):
    if(test_results[result] == -1):
        test_results[result] = 0 
    

print(test_results.shape)

print("Compiling Test Results")

print(len(test_names))

kept_names = []

# for k in range(len(test_names)):
#     name = test_names[k]
#     first = name.split("_")
        
#     if first[0] != "patient":
#         temp = test_names.pop[k]
#         k = k - 1
#     nomat = first[3].split(".")
#     test_names[k] = str(first[0]) + "_" + str(first[1]) + "_" + str(nomat[0])
        


data_to_submit = pd.DataFrame({
    #'id':test_names,
    'prediction':test_results
})

data_to_submit.to_csv('csv_to_submit.csv', index = False)
        
    
    
    