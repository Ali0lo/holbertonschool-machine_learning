#!/usr/bin/env python3
"""Tests a Keras neural network."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network."""
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
