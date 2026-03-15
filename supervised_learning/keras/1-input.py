#!/usr/bin/env python3
"""Builds a neural network with Keras Functional API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx (int): Number of input features.
        layers (list): Number of nodes in each layer.
        activations (list): Activation functions for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability of keeping a node for dropout.

    Returns:
        keras.Model: The built Keras model.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
