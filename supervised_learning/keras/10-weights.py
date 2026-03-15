#!/usr/bin/env python3
"""Save and load Keras model weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves the weights of a Keras model to a file.

    Parameters:
    - network: Keras model whose weights should be saved
    - filename: path to save the weights
    - save_format: format to save ('keras' for .h5, 'tf' for TensorFlow SavedModel)
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads weights into a Keras model from a file.

    Parameters:
    - network: Keras model into which to load the weights
    - filename: path to the saved weights file
    """
    network.load_weights(filename)
