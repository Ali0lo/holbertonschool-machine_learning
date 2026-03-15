#!/usr/bin/env python3
"""Test a Keras neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network and returns the loss and accuracy.

    Parameters:
    - network: Keras model to test
    - data: input data to test the model with
    - labels: correct one-hot labels of the data
    - verbose: whether to print the testing progress

    Returns:
    - loss, accuracy: the loss and accuracy on the test data
    """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return loss, accuracy
