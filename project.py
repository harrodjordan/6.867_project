import models 
import keras
import sklearn 
import os
import matplotlib.pyplot as plt 
import feature_extraction

data, labels, name_list = feature_extraction.raw_data(two_cat = True)

rf = models.RandomForest()
rf.train(data, labels)
sc = rf.test(data, labels)

print(sc)
print(name_list)