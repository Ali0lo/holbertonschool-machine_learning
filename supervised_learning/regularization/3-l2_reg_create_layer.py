#!/usr/bin/env python3
"""Create a TensorFlow layer with L2 regularization."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create and return a Dense layer output with L2 regularization."""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"
        ),
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )
    return layer(prev)
