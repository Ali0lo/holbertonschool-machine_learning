#!/usr/bin/env python3
"""ResNet-50 architecture"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

    Returns:
        A keras model
    """
    init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    X = K.layers.Conv2D(
        64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(7, 7),
        padding='valid'
    )(X)

    X = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=init
    )(X)

    model = K.models.Model(inputs=X_input, outputs=X)
    return model
    