#!/usr/bin/env python3
"""Creates a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network."""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init
    )
    Z = dense(prev)

    gamma = tf.Variable(
        initial_value=tf.ones((n,)),
        trainable=True
    )
    beta = tf.Variable(
        initial_value=tf.zeros((n,)),
        trainable=True
    )

    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )

    return activation(Z_norm)
