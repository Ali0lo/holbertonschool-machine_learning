#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(1, self.L + 1):
            nodes_prev = nx if l == 1 else layers[l - 2]
            nodes_curr = layers[l - 1]
            self.weights["W" + str(l)] = (
                np.random.randn(nodes_curr, nodes_prev) * np.sqrt(2 / nodes_prev)
            )
            self.weights["b" + str(l)] = np.zeros((nodes_curr, 1))
