#!/usr/bin/env python3
"""Module that defines a deep neural network for binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Class constructor."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx
        for layer in range(1, self.L + 1):
            if not isinstance(layers[layer - 1], int) or layers[layer - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights["W" + str(layer)] = (
                np.random.randn(layers[layer - 1], prev) * np.sqrt(2 / prev)
            )
            self.weights["b" + str(layer)] = np.zeros((layers[layer - 1], 1))
            prev = layers[layer - 1]
