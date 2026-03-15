#!/usr/bin/env python3
"""Module to create a batch normalization layer in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    - prev: tensor, activated output of the previous layer
    - n: number of nodes in the new layer
    - activation: activation function to apply after batch normalization

    Returns:
    - output: tensor, activated output of the layer with batch normalization
    """
    tf.random.set_seed(0)
    # Dense layer with VarianceScaling initializer
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        use_bias=False
    )(prev)

    # Batch normalization layer with trainable gamma and beta
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,  # beta
        scale=True    # gamma
    )(dense)

    # Apply activation function
    output = activation(batch_norm)

    return output
