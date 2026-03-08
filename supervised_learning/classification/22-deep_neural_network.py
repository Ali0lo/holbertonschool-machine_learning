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

    def cost(self, Y, A):
        """Calculates the cost using logistic regression."""
        m = Y.shape[1]
        cost = -1 / m * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
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
        """Trains the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
EOFcat > 22-deep_neural_network.py << 'EOF'
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

    def cost(self, Y, A):
        """Calculates the cost using logistic regression."""
        m = Y.shape[1]
        cost = -1 / m * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
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
        """Trains the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
