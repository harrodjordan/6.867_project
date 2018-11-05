import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def raw_data() :

	N = 11500
	d = 180

	df = pd.read_csv('data.csv', sep=',', header=None, low_memory=False)

	dat = df.values

	name_list = dat[:,0]

	dat_vals = dat[1:,1:-1].astype(np.float)

	labels = dat[1::,-1].astype(np.int)

	return dat_vals, labels

# sp = np.fft.fft(dat_vals)
# freq = np.fft.fftfreq(dat_vals.shape[-1])

# plt.plot(freq, sp[1,:].real)
# plt.show()
