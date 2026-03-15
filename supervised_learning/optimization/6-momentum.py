#!/usr/bin/env python3
"""Module to create a gradient descent with momentum optimizer."""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates a TensorFlow optimizer for gradient descent with momentum.

    Parameters:
    - alpha: learning rate
    - beta1: momentum weight

    Returns:
    - optimizer: a TensorFlow optimizer instance
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
