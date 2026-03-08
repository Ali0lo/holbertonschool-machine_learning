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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for layer in range(1, self.__L + 1):
            if not isinstance(layers[layer - 1], int) or layers[layer - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W" + str(layer)] = (
                np.random.randn(layers[layer - 1], prev) * np.sqrt(2 / prev)
            )
            self.__weights["b" + str(layer)] = np.zeros((layers[layer - 1], 1))
            prev = layers[layer - 1]

    @property
    def L(self):
        """Getter for L."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W" + str(layer)]
            b = self.__weights["b" + str(layer)]
            A_prev = self.__cache["A" + str(layer - 1)]
            Z = np.dot(W, A_prev) + b
            self.__cache["A" + str(layer)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache
