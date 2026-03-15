#!/usr/bin/env python3
"""Deep Neural Network module for multiclass classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            layer_key_W = "W{}".format(i + 1)
            layer_key_b = "b{}".format(i + 1)

            if i == 0:
                self.__weights[layer_key_W] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[layer_key_W] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            self.__weights[layer_key_b] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """number of layers"""
        return self.__L

    @property
    def cache(self):
        """activated outputs cache"""
        return self.__cache

    @property
    def weights(self):
        """weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation using softmax on output layer"""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]
            Z = np.matmul(W, A_prev) + b

            if i == self.__L:
                Z_shift = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_shift)
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates cost using multiclass cross-entropy"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A + 1e-8)) / m

    def evaluate(self, X, Y):
        """Evaluates network predictions for multiclass classification"""
        A, _ = self.forward_prop(X)
        m = A.shape[1]
        classes = A.shape[0]

        pred_idx = np.argmax(A, axis=0)
        predictions = np.zeros((classes, m))
        predictions[pred_idx, np.arange(m)] = 1

        return predictions, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        for i in range(self.__L, 0, -1):
            A = cache["A{}".format(i)]
            A_prev = cache["A{}".format(i - 1)]

            if i == self.__L:
                dZ = A - Y
            else:
                W_next = weights_copy["W{}".format(i + 1)]
                dZ = np.matmul(W_next.T, dZ) * (A * (1 - A))

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W{}".format(i)] = (
                weights_copy["W{}".format(i)] - alpha * dW
            )
            self.__weights["b{}".format(i)] = (
                weights_copy["b{}".format(i)] - alpha * db
            )

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
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

            if i % step == 0 or i == iterations:
                c = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    steps.append(i)
                    costs.append(c)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
