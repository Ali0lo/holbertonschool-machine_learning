#!/usr/bin/env python3
"""DenseNet-121 architecture"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture

    Args:
        growth_rate: growth rate
        compression: compression factor

    Returns:
        keras model
    """
    init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Initial BN + ReLU + Conv
    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        2 * growth_rate,
        (7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    nb_filters = 2 * growth_rate

    # Dense Block 1 (6 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2 (12 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3 (24 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4 (16 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final layers
    X = K.layers.AveragePooling2D(
        (7, 7),
        strides=(1, 1),
        padding='valid'
    )(X)

    X = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=init
    )(X)

    model = K.models.Model(inputs=X_input, outputs=X)
    return model
