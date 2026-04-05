#!/usr/bin/env python3
"""
Module containing the function lenet5
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras.

    Parameters:
    - X: K.Input of shape (m, 28, 28, 1) containing the input images

    Returns:
    A K.Model compiled to use Adam optimization and accuracy metrics.
    """
    # Define the weight initializer
    init = K.initializers.HeNormal(seed=0)

    # Layer 1: Convolutional (6 kernels, 5x5, same padding, ReLU)
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # Layer 2: Max Pooling (2x2 kernels, 2x2 strides)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Layer 3: Convolutional (16 kernels, 5x5, valid padding, ReLU)
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    # Layer 4: Max Pooling (2x2 kernels, 2x2 strides)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the pooling output for the Fully Connected layers
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Fully Connected (120 nodes, ReLU)
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flatten)

    # Layer 6: Fully Connected (84 nodes, ReLU)
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    # Layer 7: Fully Connected Softmax Output (10 nodes)
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(fc2)

    # Construct the model
    model = K.Model(inputs=X, outputs=output)

    # Compile the model
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
