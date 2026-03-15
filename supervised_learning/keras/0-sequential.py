#!/usr/bin/env python3
"""Builds a neural network with Keras (Sequential API)."""

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
        keras.Model: The built Keras Sequential model.
    """
    model = K.models.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_shape=(nx,)
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
