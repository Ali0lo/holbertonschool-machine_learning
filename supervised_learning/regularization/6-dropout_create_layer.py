#!/usr/bin/env python3
"""Create TensorFlow layer using dropout."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a dense layer followed by dropout."""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer='he_normal'
    )(prev)

    if training:
        layer = tf.keras.layers.Dropout(rate=1 - keep_prob)(layer)

    return layer
