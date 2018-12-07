import scipy.signal as sig
import pandas as pd 
import numpy as np 
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


data = pd.read_csv('data.csv', sep=',', header=None, low_memory=False)


data = data.values 

name_list = data[1:,0]

data = data[1:,1:-1].astype(np.float)

#name_list = data[1:,0]
print(name_list)

if not os.path.isdir("/home/paperspace/6.867_project/eeg_spec"): os.mkdir("/home/paperspace/6.867_project/eeg_spec")

images = []

fs = 173.61

plt.ioff()

dir = '/home/paperspace/6.867_project/eeg_spec/'

for i in range(11500):

    f, t, Sxx = sig.spectrogram(data[i, :], fs)
    fig = plt.pcolormesh(t, f, Sxx) 
    name = name_list[i].split('.')
    if len(name) == 2: name.append('')
    file_name = dir + name[0] + name[1] + name[2] + '.jpg'
    plt.savefig(file_name)
    plt.close()



