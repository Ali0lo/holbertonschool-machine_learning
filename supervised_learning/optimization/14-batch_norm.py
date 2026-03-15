#!/usr/bin/env python3
"""Batch normalization layer creation in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    - prev: activated output of the previous layer (NumPy array or tensor)
    - n: number of nodes in the new layer
    - activation: activation function to apply after batch normalization

    Returns:
    - output: tensor, activated output of the layer with batch normalization
    """
    # Convert input to tensor if needed
    if not isinstance(prev, tf.Tensor):
        prev = tf.convert_to_tensor(prev, dtype=tf.float32)

    # Dense layer with VarianceScaling initializer and no bias
    a = 'fan_avg'
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode=a),
        use_bias=False
    )(prev)

    # Batch normalization with trainable gamma and beta
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,  # beta
        scale=True    # gamma
    )(dense, training=True)  # ensure batch statistics are used

    # Apply activation
    output = activation(batch_norm)

    return output
