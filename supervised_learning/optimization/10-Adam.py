#!/usr/bin/env python3
"""Module to create an Adam optimizer in TensorFlow."""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates a TensorFlow optimizer using the Adam algorithm.

    Parameters:
    - alpha: learning rate
    - beta1: weight for the first moment (momentum)
    - beta2: weight for the second moment (RMSProp)
    - epsilon: small number to avoid division by zero

    Returns:
    - optimizer: a TensorFlow Adam optimizer instance
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
    return optimizer
