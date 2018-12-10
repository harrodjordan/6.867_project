import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def raw_data(two_cat=False) :

    N = 11500
    d = 180

    df = pd.read_csv('data.csv', sep=',', header=None, low_memory=False)

    dat = df.values

    name_list = dat[1:,0]

    dat_vals = dat[1:,1:-1].astype(np.float)

    labels = dat[1::,-1].astype(np.int)

    if two_cat :
        labels = (labels==1).astype(float)

    return dat_vals, labels, name_list

def features(): 
    
    #data, labels, name_list = raw_data()
    data, bin_labels, name_list = raw_data(two_cat=True)
    
    sigs = np.std(data,axis=1)

    length = 178

    energy = np.sum(np.square(data, axis=1), axis=1)

    L = 0
    
    for j in range(length-1):
        L = L + (data[j+1,:] - data[j,:]);

    line_length = np.sum(L/length, axis=1);


    entropy = np.sum(np.square(data, axis=1)*np.log10(np.square(data, axis=1), axis=1), axis=1);
    
    feat = [sigs, energy, line_length, entropy]


    return feat
    
    
