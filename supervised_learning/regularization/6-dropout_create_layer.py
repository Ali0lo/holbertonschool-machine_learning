#!/usr/bin/env python3
"""Create TensorFlow layer using dropout."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a dense layer followed by dropout."""
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"
        )
    )
    x = dense(prev)
    drop = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return drop(x, training=training)
