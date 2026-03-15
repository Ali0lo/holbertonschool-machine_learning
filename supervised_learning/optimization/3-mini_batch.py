#!/usr/bin/env python3
"""Module to create mini-batches from data for mini-batch gradient descent."""


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from input data X and labels Y.

    Parameters:
    - X: numpy.ndarray of shape (m, nx) representing input data
         m is the number of data points
         nx is the number of features
    - Y: numpy.ndarray of shape (m, ny) representing labels
         m is the number of data points
         ny is the number of classes
    - batch_size: number of data points in each batch

    Returns:
    - mini_batches: list of tuples (X_batch, Y_batch)
    """
    # Import shuffle_data function
    shuffle_data = __import__('2-shuffle_data').shuffle_data

    # Shuffle the data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    # Partition into batches
    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
