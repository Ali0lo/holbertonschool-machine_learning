#!/usr/bin/env python3
""" This module will define a class named DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """ Initialize class """
        # validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # validate layers
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        # private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx

        for i, nodes in enumerate(layers):

            # validate nodes
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            # He initialization
            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )

            # bias initialization
            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

            prev = nodes

    # getters
    @property
    def L(self):
        """ getter for L """
        return self.__L

    @property
    def cache(self):
        """ getter for cache """
        return self.__cache

    @property
    def weights(self):
        """ get for weihts """
        return self.__weights

    def forward_prop(self, X):
        """ forward propagation """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):

            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]

            A_prev = self.__cache["A{}".format(layer-1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(layer)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the logistic regression cost"""

        m = Y.shape[1]

        cost = -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
            ) / m

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""

        AL, _ = self.forward_prop(X)
        prediction = np.where(AL >= 0.5, 1, 0)
        cost = self.cost(Y, AL)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        dZ = cache["A" + str(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache["A" + str(layer - 1)]
            W = self.__weights["W" + str(layer)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))
            self.__weights["W" + str(layer)] -= alpha * dW
            self.__weights["b" + str(layer)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network"""
        
        # Input validation
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        # Training loop
        for i in range(iterations):
            AL, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        
        # Evaluate after training
        return self.evaluate(X, Y)
