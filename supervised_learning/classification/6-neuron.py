#!/usr/bin/env python3
""" This module will define a class named Neuron """
import numpy as np


class Neuron():
    """ Class for implementing neuron in NN """
    def __init__(self, nx):
        """ initialize class """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Return W """
        return self.__W

    @property
    def b(self):
        """ Return b """
        return self.__b

    @property
    def A(self):
        """ Return A"""
        return self.__A

    def forward_prop(self, X):
        """ Function for conducting forward propagation """
        z = self.__W @ X + self.__b
        self.__A = 1 / (1 + np.e**(-z))
        return self.__A

    def cost(self, Y, A):
        """ Cost of the model (based on logistic reg) """
        J = np.sum(
            -Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
            ) / A.shape[1]
        return J

    def evaluate(self, X, Y):
        """ evaluate neuron """
        A = self.forward_prop(X)
        Y_pred = np.where(A >= 0.5, 1, 0)
        return Y_pred, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ apply gradient descent algorithm """
        dz = A - Y
        m = dz.shape[1]
        self.__W -= (alpha * dz @ X.T) / m
        self.__b -= (alpha * np.sum(dz)) / m

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Train neuron """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        elif iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        elif alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        
        self.evaluate(X, Y)
