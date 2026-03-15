#!/usr/bin/env python3
"""Module that defines a deep neural network for multiclass classification."""

import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """Class constructor."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
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

    @property
    def activation(self):
        """Getter for activation function."""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W" + str(layer)]
            b = self.__weights["b" + str(layer)]
            A_prev = self.__cache["A" + str(layer - 1)]
            Z = np.dot(W, A_prev) + b
            if layer == self.__L:
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache["A" + str(layer)] = t / np.sum(t,
                                                            axis=0,
                                                            keepdims=True)
            else:
                if self.__activation == 'sig':
                    self.__cache["A" + str(layer)] = 1 / (1 + np.exp(-Z))
                else:
                    self.__cache["A" + str(layer)] = np.tanh(Z)

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ cost function """
        m = Y.shape[1]
        A = np.clip(A, 1e-7, 1 - 1e-7)
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Convert to one-hot
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
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
            if self.__activation == 'sig':
                dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))
            else:
                dZ = np.dot(W.T, dZ) * (1 - A_prev ** 2)
            self.__weights["W" + str(layer)] -= alpha * dW
            self.__weights["b" + str(layer)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and i % step == 0:
                costs.append(cost)
                steps.append(i)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(steps, costs, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format."""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object from a file."""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
