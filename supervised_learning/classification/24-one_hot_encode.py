#!/usr/bin/env python3
""" This module will define a function for one hot encoding """
import numpy as np


def one_hot_encode(Y, classes):
    """ Converts a numeric label vector into a one-hot matrix """
    try:
        Y = np.array(Y, dtype=int)  # ensure integer type
        m = Y.shape[0]
        if m == 0 or classes <= 0:
            return None
        
        # Initialize the one-hot matrix with zeros
        one_hot = np.zeros((classes, m))
        
        # Set the corresponding element to 1 for each example
        one_hot[Y, np.arange(m)] = 1
        
        return one_hot
    
    except Exception:
        return None
