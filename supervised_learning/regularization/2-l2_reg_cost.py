#!/usr/bin/env python3
"""L2 regularized cost for Keras model."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate total cost per layer including L2 regularization."""
    reg_terms = [cost]
    for loss in model.losses:
        reg_terms.append(loss)
    return tf.stack(reg_terms)
