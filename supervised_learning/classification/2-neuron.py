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
        self.__A = 1 / (1 + e**(-z))
        return self.__A
