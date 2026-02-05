#!/usr/bin/env python3
"""Module for numpy operations"""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    slices = [slice(None)] * matrix.ndim
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
