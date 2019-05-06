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

from Unet2D import *
from keras import callbacks
from keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt
import os
#from model2D import *
#from model2D import tindex,vindex
#from train2D_5 import *
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


n1, n2 = 96,96
params = {'batch_size': 1,
          'dim':(n1,n2),
          'n_channels': 1,
          'shuffle': True}

tdpath = 'images_1/train_gen/seis/'
tfpath = 'images_1/train_gen/labels/'

vdpath = 'images_1/validation/seis/'
vfpath = 'images_1/validation/labels/'
#tindex=8588
#vindex=2000
import parameters as pa
tdata_IDs = range(pa.tindex)
vdata_IDs = range(pa.vindex)
training_generator   = DataGenerator(dpath=tdpath,fpath=tfpath,data_IDs=tdata_IDs,**params)
validation_generator = DataGenerator(dpath=vdpath,fpath=vfpath,data_IDs=vdata_IDs,**params)


#model=unet()
K.set_image_data_format('channels_last')
'''
model_name = 'fault'
model_dir     = os.path.join('check', model_name)
csv_fn        = os.path.join(model_dir, 'train_log.csv')
checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}.hdf5')


checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=False)
'''
model_dir  = 'check/fault/'
csv_path = 'check/train_log4.csv'
model = unet()
# model_name = 'fault'
# checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}.hdf5')

checkpointer = callbacks.ModelCheckpoint('images/train_gen/my_unet_generator_seq4.hdf5', verbose=1,
                                             save_best_only=False)
csv_logger  = callbacks.CSVLogger(csv_path, append=True, separator=';')
tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=2,
                                        write_graph=True, write_grads=True, write_images=True)
#csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')
#tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=2,
                                        #write_graph=True, write_grads=True, write_images=True)
history = model.fit_generator(
                        generator=training_generator,
                        validation_data=validation_generator,
                        epochs=15,verbose=1,callbacks=[checkpointer, csv_logger, tensorboard])
model.summary()
print(history)


model_json = model.to_json()
with open("model/model2_4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model2_4.h5")

print(history.history.keys())
fig = plt.figure(figsize=(10,6))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xlabel('Epoch',fontsize=20)
plt.legend(['train', 'validation'], loc='center right',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
#plt.show()

# summarize history for loss
fig = plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xlabel('Epoch',fontsize=20)
plt.legend(['train', 'validation'], loc='center right',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
plt.show()


