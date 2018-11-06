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
