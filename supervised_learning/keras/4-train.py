#!/usr/bin/env python3
"""Train a keras model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Trains the model using mini-batch gradient descent"""

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
