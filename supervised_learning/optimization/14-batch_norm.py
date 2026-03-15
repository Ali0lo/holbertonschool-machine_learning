#!/usr/bin/env python3
"""Creates a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Returns the activated output of a batch-normalized dense layer."""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    Z = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )(prev)

    Z_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(Z)

    return tf.keras.layers.Activation(activation)(Z_norm)
