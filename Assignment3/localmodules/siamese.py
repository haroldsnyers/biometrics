#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:15:55 2019

@author: dvdm
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Conv2D, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MaxPooling2D, Lambda, Flatten, Dense, BatchNormalization
from tensorflow.keras import backend as K

import numpy as np
import numpy.random as rng


def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(label, ED):
    margin = 1
    # note: the images are scaled between 0 and 1
    return K.mean((1- label) * 0.5*K.square(ED) + label * 0.5* K.square(K.maximum(margin - ED, 0)))

def accuracy(y_true, y_pred):
    y_true_bool = K.cast(y_true, dtype='bool')
    return K.mean(K.equal(y_true_bool, y_pred > 0.5))

def create_shared_network(input_shape):
    
    # small CNN
    # nb_filter = [16, 8]
    # larger CNN
    nb_filter = [128, 64]
    kernel_size = 3

    model = Sequential()
    model.add(Conv2D(filters=nb_filter[0], kernel_size=kernel_size, activation='relu', 
                     input_shape=input_shape))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(filters=nb_filter[1], kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    return model

def create_siamese_model(input_shape):
    shared_network = create_shared_network(input_shape)
    
    input_top = Input(shape=input_shape)
    input_bottom = Input(shape=input_shape)
    output_top = shared_network(input_top)
    output_bottom = shared_network(input_bottom)
    
    distance = Lambda(euclidean_distance, output_shape=(1,))([output_top, output_bottom])

    model = Model(inputs=[input_top, input_bottom], outputs=distance)
    
    return shared_network, model

def get_siamese_paired_data(X, y, total_sample_size = 1000):
    """
        Create batch of n pairs, half same class, half different class
    """
    
    assert len(X.shape) == 4, "Expected format for X: (n_samples, height, width, channels)"
    
    n_samples, height, width, channels = X.shape
    n_classes = np.unique(y).shape[0]
    
    
    # randomly sample classes 
    categories = rng.choice(n_classes,size=(2*total_sample_size,),replace=True)

    genuine_pairs=np.zeros((total_sample_size, 2, width, height, channels))
    imposter_pairs=np.zeros((total_sample_size, 2, width, height, channels))

    # initialize vector for the targets
    # make first half genuine pairs (0), second half imposter pairs (1)
    targets=np.zeros((2*total_sample_size),)
    targets[total_sample_size:] = 1

    for i, category in enumerate(categories):
        same_category_indices = np.where(y == category)[0]

        if i < total_sample_size:
            idx_1 = rng.randint(0, len(same_category_indices))
            genuine_pairs[i,0,:] = X[same_category_indices[idx_1]]
            idx_2 = rng.randint(0, len(same_category_indices))
            genuine_pairs[i,1,:] = X[same_category_indices[idx_2]] 

        else:
            diff_category_indices = np.where(y != category)[0]

            idx_1 = rng.randint(0, len(same_category_indices))
            imposter_pairs[i-total_sample_size,0,:] = X[same_category_indices[idx_1]]
            idx_2 = rng.randint(0, len(diff_category_indices))
            imposter_pairs[i-total_sample_size,1,:] = X[diff_category_indices[idx_2]] 

    return np.concatenate([genuine_pairs, imposter_pairs] , axis=0)/255, targets
