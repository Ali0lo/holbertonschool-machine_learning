#!/usr/bin/env python3
"""Module to create a learning rate decay operation using TensorFlow."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    TensorFlow learning rate decay operation using inverse time decay.

    Parameters:
    - alpha: original learning rate
    - decay_rate: weight used to determine the rate of decay
    - decay_step: number of gradient descent passes before decaying alpha

    Returns:
    - learning_rate: a TensorFlow callable for the decayed learning rate
    """
    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return learning_rate
