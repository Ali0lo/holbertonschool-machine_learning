#!/usr/bin/env python3
"""Module that defines a NeuralNetwork with one hidden layer."""
import numpy as np


class NeuralNetwork:
    """Neural network with one hidden layer for binary classification."""

    def __init__(self, nx, nodes):
        """ Initialize NeuralNetwork """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2."""
        return self.__A2

    def forward_prop(self, X):
        """ Forward propagation """
        z1 = self.__W1 @ X + self.__b1
        self.__A1 = 1 / (1 + np.e**(-z1))
        z2 = self.__W2 @ self.__A1 + self.__b2
        self.__A2 = 1 / (1 + np.e**(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Cost of the model (based on logistic reg) """
        J = np.sum(
            -Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
            ) / A.shape[1]
        return J

    def evaluate(self, X, Y):
        """ evaluate NN """
        A1, A2 = self.forward_prop(X)
        Y_pred = np.where(A2 >= 0.5, 1, 0)
        return Y_pred, self.cost(Y, A2)
