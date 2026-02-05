#!/usr/bin/env python3
"""Module for matrix operations"""


def add_matrices(mat1, mat2):
    """Adds two matrices of any dimension"""
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            sub_result = add_matrices(mat1[i], mat2[i])
            if sub_result is None:
                return None
            result.append(sub_result)
        return result
    elif not isinstance(mat1, list) and not isinstance(mat2, list):
        return mat1 + mat2
    else:
        return None
