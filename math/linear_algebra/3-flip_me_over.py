#!/usr/bin/env python3
"""Module to transpose a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix

    Args:
        matrix: A 2D matrix (list of lists)

    Returns:
        A new matrix that is the transpose of the input matrix
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
