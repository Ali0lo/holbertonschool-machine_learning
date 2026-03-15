#!/usr/bin/env python3
"""Save and load Keras model configuration in JSON using only Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves the model's architecture (configuration) to a JSON file.

    Parameters:
    - network: Keras model whose configuration should be saved
    - filename: path to save the JSON file
    """
    # Get model configuration as JSON string
    config_json = network.to_json()

    # Write JSON string to file
    with open(filename, 'w') as f:
        f.write(config_json)


def load_config(filename):
    """
    Loads a model from a JSON configuration file.

    Parameters:
    - filename: path to the JSON file containing model configuration

    Returns:
    - A new Keras model with the loaded architecture (uncompiled)
    """
    # Read JSON string from file
    with open(filename, 'r') as f:
        config_json = f.read()

    # Recreate the model from JSON
    model = K.models.model_from_json(config_json)
    return model
