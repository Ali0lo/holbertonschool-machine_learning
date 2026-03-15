#!/usr/bin/env python3
"""Make predictions using a Keras neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes predictions using a Keras neural network.

    Parameters:
    - network: Keras model to make predictions with
    - data: input data to predict
    - verbose: whether to print the prediction process

    Returns:
    - The predictions of the network on the input data
    """
    predictions = network.predict(data, verbose=verbose)
    return predictions
