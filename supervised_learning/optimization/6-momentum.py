#!/usr/bin/env python3
"""Creates a Momentum optimization operation in TensorFlow."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Returns the Momentum optimizer."""
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
