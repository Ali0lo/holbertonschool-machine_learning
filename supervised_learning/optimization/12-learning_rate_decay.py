#!/usr/bin/env python3
"""Creates a TensorFlow inverse time decay learning rate schedule."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Returns an inverse time decay learning rate operation."""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
