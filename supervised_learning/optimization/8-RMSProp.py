#!/usr/bin/env python3
"""Creates an RMSProp optimization operation in TensorFlow."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Returns the RMSProp optimizer."""
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
