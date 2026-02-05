#!/usr/bin/env python3
"""Module for matrix operations"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""

    def get_shape(mat):
        """Get the shape of a matrix"""
        if not isinstance(mat, list):
            return []
        if len(mat) == 0:
            return [0]
        return [len(mat)] + get_shape(mat[0])

    def deep_copy(mat):
        """Create a deep copy of a matrix"""
        if not isinstance(mat, list):
            return mat
        return [deep_copy(elem) for elem in mat]

    def cat_recursive(m1, m2, current_depth):
        """Recursively concatenate matrices"""
        if not isinstance(m1, list) or not isinstance(m2, list):
            return None

        if current_depth == axis:
            return deep_copy(m1) + deep_copy(m2)

        if len(m1) != len(m2):
            return None

        result = []
        for i in range(len(m1)):
            sub_result = cat_recursive(m1[i], m2[i], current_depth + 1)
            if sub_result is None:
                return None
            result.append(sub_result)
        return result

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    if len(shape1) != len(shape2):
        return None

    if axis < 0 or axis >= len(shape1):
        return None

    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    return cat_recursive(mat1, mat2, 0)
