import models 
import sklearn 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import feature_extraction
from sklearn.model_selection import cross_val_score
import numpy as np
import torch
import models
from models import train_test_split
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.decomposition import PCA
import seaborn as sns

seq_length = 178

#history = keras.callbacks.History()

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

optim = keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=optim,
              metrics=['acc'])


data, labels, name_list = feature_extraction.raw_data(two_cat = True)
#data = feature_extraction.features()
#print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, name_list, test_size = 0.2)
print(X_train.shape)



X_train = X_train[:,np.newaxis,:]
X_train = np.swapaxes(X_train, 1, 2)

X_test = X_test[:,np.newaxis,:]
X_test = np.swapaxes(X_test, 1, 2)

history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_split=0.1)
score = model.evaluate(X_test, y_test, batch_size=16)
print(score)

from sklearn.metrics import roc_curve
y_pred_keras = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('CNN ROC')
plt.close()

print(history.history)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Training and Validation Loss')
plt.close()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Training and Validation Accuracy')
plt.close()

from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[10].reshape(1,178,1))
 

from sklearn.metrics import confusion_matrix
Y_prediction = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_prediction,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt="d");
plt.savefig('Confusion Matrix')
plt.close()
