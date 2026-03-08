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
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
