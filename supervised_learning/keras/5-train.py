#!/usr/bin/env python3
"""Trains a Keras model using mini-batch gradient descent with validation."""


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network: The model to train.
        data: numpy.ndarray of shape (m, nx) with input data.
        labels: One-hot numpy.ndarray of shape (m, classes) with labels.
        batch_size: Size of mini-batches.
        epochs: Number of epochs.
        validation_data: Data to validate the model with, if not None.
        verbose: Whether to print training output.
        shuffle: Whether to shuffle batches every epoch.

    Returns:
        The History object generated after training.
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )
