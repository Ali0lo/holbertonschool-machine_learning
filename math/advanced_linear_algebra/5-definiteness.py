#!/usr/bin/env python3
"""Module for calculating the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix

    Args:
        matrix: a numpy.ndarray of shape (n, n) whose definiteness
                should be calculated

    Returns:
        The string Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or Indefinite
        if the matrix is positive definite, positive semi-definite,
        negative semi-definite, negative definite, or indefinite
        respectively. Returns None if matrix does not fit any category
        or is not a valid matrix.

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is 2D and not empty
    if matrix.ndim != 2 or matrix.size == 0:
        return None

    # Check if matrix is square
    n, m = matrix.shape
    if n != m:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check definiteness based on eigenvalues
    # Use small tolerance for numerical comparisons
    tol = 1e-10

    if np.all(eigenvalues > tol):
        return "Positive definite"
    elif np.all(eigenvalues >= -tol):
        return "Positive semi-definite"
    elif np.all(eigenvalues < -tol):
        return "Negative definite"
    elif np.all(eigenvalues <= tol):
        return "Negative semi-definite"
    else:
        return "Indefinite"
