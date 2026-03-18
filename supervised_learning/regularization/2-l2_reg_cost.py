#!/usr/bin/env python3
"""L2 regularization cost module."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate total cost per layer including L2 regularization losses."""
    return tf.stack([cost] + model.losses)
