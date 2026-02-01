#!/usr/bin/env python3
"""Module to calculate the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix
    
    Args:
        matrix: A matrix (list of lists)
    
    Returns:
        A list of integers representing the shape
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
