#!/usr/bin/env python3
""" This module will define a class named Neuron """
import numpy as np


class NeuralNetwork():
    """ This class will define a neural network with one hidden layer """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(1, nx)
        self.__b1= 0
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nx)
        self.__b2= 0
        self.__A2 = 0
