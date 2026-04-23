#!/usr/bin/env python3
"""Dense Block for DenseNet"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block

    Args:
        X: output from previous layer
        nb_filters: number of filters in X
        growth_rate: growth rate
        layers: number of layers in the block

    Returns:
        concatenated output and updated number of filters
    """
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        # Bottleneck layer
        BN1 = K.layers.BatchNormalization(axis=3)(X)
        RELU1 = K.layers.Activation('relu')(BN1)
        CONV1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=init
        )(RELU1)

        # 3x3 convolution
        BN2 = K.layers.BatchNormalization(axis=3)(CONV1)
        RELU2 = K.layers.Activation('relu')(BN2)
        CONV2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=init
        )(RELU2)

        # Concatenate with input
        X = K.layers.Concatenate(axis=3)([X, CONV2])

        # Update filter count
        nb_filters += growth_rate

    return X, nb_filters
