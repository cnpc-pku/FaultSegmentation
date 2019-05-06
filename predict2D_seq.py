# -*- coding: utf-8 -*-
"""

# Project Name:     Fault Prediction
# File Name:        train2D-2
# Date:             04/10/2019 10:45 AM
# Using IDE:        PyCharm Community Edition
# Author:           Donglin Zhu
# E-mail:           zhudonglin@cnpc.com.cn
# Copyright (c) 2019, All Rights Reserved.

This is a temporary script file.
"""


from keras import callbacks
from keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import *

import os
#from model2D import *
#from model2D import tindex,vindex
#from train2D import DataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class DataGenerator(keras.utils.Sequence):
    'Generates data for keras'

    def __init__(self, dpath, fpath, data_IDs, batch_size=1, dim=(96,96),
                 n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dpath = dpath
        self.fpath = fpath
        self.batch_size = batch_size
        self.data_IDs = data_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        # Generate indexes of the batch
        bsize = self.batch_size
        indexes = self.indexes[index * bsize:(index + 1) * bsize]

        # Find list of IDs
        data_IDs_temp = [self.data_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(data_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((1, *self.dim, self.n_channels), dtype=np.single)
        Y = np.zeros((1, *self.dim, self.n_channels), dtype=np.single)
        gx = np.load(self.dpath + str(data_IDs_temp[0]) + '.npy')#, dtype=np.single
        fx = np.load(self.fpath + str(data_IDs_temp[0]) + '.npy')
        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)
        gx = gx - np.min(gx)
        gx = gx / np.max(gx)
        gx = gx * 255
        # Generate data
        #for i in range(4):
        for i in range(1):
            X[i,] = np.reshape(gx, (*self.dim, self.n_channels))
            Y[i,] = np.reshape(fx, (*self.dim, self.n_channels))
        return X, Y

class DataGenerator_test(keras.utils.Sequence):
    'Generates data for keras'

    def __init__(self, dpath, data_IDs, batch_size=1, dim=(96,96),
                 n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dpath = dpath

        self.batch_size = batch_size
        self.data_IDs = data_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        # Generate indexes of the batch
        bsize = self.batch_size
        indexes = self.indexes[index * bsize:(index + 1) * bsize]

        # Find list of IDs
        data_IDs_temp = [self.data_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(data_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((1, *self.dim, self.n_channels), dtype=np.single)

        gx = np.load(self.dpath + str(data_IDs_temp[0]) + '.npy')#, dtype=np.single

        gx = np.reshape(gx, self.dim)

        gx = gx - np.min(gx)
        gx = gx / np.max(gx)
        gx = gx * 255
        # Generate data
        #for i in range(4):
        for i in range(1):
            X[i,] = np.reshape(gx, (*self.dim, self.n_channels))

        return X


n1, n2 = 96,96
params = {'batch_size': 1,
          'dim':(n1,n2),
          'n_channels': 1,
          'shuffle': True}
tdpath = 'images_1/test/seis/'
tfpath='images_1/test/labels/'
import parameters as pa
tdata_IDs = range(pa.testindex)

test_generator   = DataGenerator_test(dpath=tdpath,data_IDs=tdata_IDs,**params)
eva_generator=DataGenerator(dpath=tdpath,fpath=tfpath,data_IDs=tdata_IDs,**params)

# load json and create model
json_file = open('model/model2_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("images/train_gen/my_unet_generator_seq4.hdf5")
print("Loaded model from disk")
#model = unet()
#model.load_weights('F:/PyCharm/Practice3/images/train_gen/my_unet_generator.hdf5')
#x = np.reshape(gx,(1,96,64,1))

from keras.models import *
from keras.optimizers import *

def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits,
    # Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

loaded_model.compile(optimizer=Adam(lr=1e-4), loss=cross_entropy_balanced, metrics=['accuracy'])
Eva=loaded_model.evaluate_generator(eva_generator,verbose=1)
print("Loss = ",Eva[0])
print("Accuracy = ",Eva[1])
'''
print(Eva.history.keys())
fig = plt.figure(figsize=(10,6))

# summarize history for accuracy
plt.plot(Eva.history['acc'])
plt.plot(Eva.history['val_acc'])
plt.title('Model accuracy',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xlabel('Epoch',fontsize=20)
plt.legend(['train', 'validation'], loc='center right',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
#plt.show()

# summarize history for loss
fig = plt.figure(figsize=(10,6))
plt.plot(Eva.history['loss'])
plt.plot(Eva.history['val_loss'])
plt.title('Model loss',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xlabel('Epoch',fontsize=20)
plt.legend(['train', 'validation'], loc='center right',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
plt.show()

#print(Y)
#i=int(input("number of seis:"))
#i=15
'''
Y = loaded_model.predict_generator(test_generator,verbose=1)
for i in range(pa.testindex):
    fig = plt.figure(figsize=(20,20))
    #inline slice
    gx,m1,m2 = np.load('images_1/test/seis/'+str(i)+'.npy'),96,96
    fx=np.load('images_1/test/labels/'+str(i)+'.npy')
    x = np.reshape(gx,(1,96,96,1))
    plt.subplot(1, 3, 1)
    imgplot1 = plt.imshow(x[0,:,:,0],cmap=plt.cm.gray,interpolation='nearest',aspect=1)
    plt.subplot(1, 3, 2)
    imgplot2 = plt.imshow(Y[i,:,:,0],cmap=plt.cm.gray,interpolation='nearest',aspect=1)#interpolation='nearest',
    plt.subplot(1, 3, 3)
    fx=np.reshape(fx,(96,96))
    imgplot3 = plt.imshow(fx,cmap=plt.cm.gray,interpolation='nearest',aspect=1)
    #plt.show()
    plt.savefig('images_1/test/predict/' + str(i) + '.jpg')
    plt.close(fig)



