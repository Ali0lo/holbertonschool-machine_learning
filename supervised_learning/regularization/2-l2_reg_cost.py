#!/usr/bin/env python3
"""Module for calculating L2 regularization cost in TensorFlow."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculate the cost of a neural network with L2 regularization.

    Args:
        cost: A tensor containing the cost of the network without L2
              regularization
        model: A Keras model that includes layers with L2 regularization

    Returns:
        A tensor containing the total cost for each layer of the network,
        accounting for L2 regularization
    """
    # Get all regularization losses from the model
    # model.losses contains the L2 regularization losses for each layer
    layer_costs = []

    for reg_loss in model.losses:
        # Add the base cost to each layer's regularization loss
        total_cost = cost + reg_loss
        layer_costs.append(total_cost)

    # Stack into a single tensor
    return tf.stack(layer_costs)
