#!/usr/bin/env python3
"""Transition layer for DenseNet"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    Densely Connected Convolutional Networks

    Args:
        X: output from previous layer
        nb_filters: number of filters in X
        compression: compression factor for DenseNet-C

    Returns:
        output of the transition layer and the new number of filters
    """
    init = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(X)

    return X, nb_filters
