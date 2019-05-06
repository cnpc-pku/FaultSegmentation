# -*- coding: utf-8 -*-
"""

# Project Name:     Fault Prediction
# File Name:        Unet2D
# Date:             04/3/2019 2:55 PM
# Using IDE:        PyCharm Community Edition
# Author:           Donglin Zhu
# E-mail:           zhudonglin@cnpc.com.cn
# Copyright (c) 2019, All Rights Reserved.

This is a temporary script file.
"""

import tensorflow.contrib.keras as tkeras
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import *
import numpy as np
#drate=0.5

def unet(pretrained_weights = None,drate=0,inputs=Input(shape=(96,96,1))):
    #inputs=Input(shape=(96,64,1))
    conv1=Conv2D(64,(3,3),activation = 'relu', padding = 'same')(inputs)#, kernel_initializer = 'he_normal'
    conv1=Dropout(drate)(conv1)
    conv1=Conv2D(64,(3,3),activation = 'relu', padding = 'same')(conv1)
    conv1 = Dropout(drate)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(drate)(conv2)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
    conv2 = Dropout(drate)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(drate)(conv3)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(drate)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(drate)(conv4)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    conv4 = Dropout(drate)(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3) #axis may need revise
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(up5)
    conv5 = Dropout(drate)(conv5)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(drate)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(up6)
    conv6 = Dropout(drate)(conv6)
    conv6 = Conv2D(128, (3,3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(drate)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = Conv2D(64, (3,3), activation='relu', padding='same')(up7)
    conv7 = Dropout(drate)(conv7)
    conv7 = Conv2D(64, (3,3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(drate)(conv7)

    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(input=inputs, output=conv8)
    #model = Model(inputs=[inputs], outputs=[conv8])
    model.compile(optimizer=Adam(lr=1e-4), loss=cross_entropy_balanced, metrics=['accuracy'])
    #plot_model(model,to_file='model/model.png')
    return model

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