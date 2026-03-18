#!/usr/bin/env python3
"""L2 regularized cost for Keras model."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate total cost including model L2 regularization losses."""
    return cost + tf.reduce_sum(model.losses)
