def summation_i_squared(n):
    """
    This function computes the sum of squares of integers from 1 to n.

    Arguments:
    n -- the stopping condition (positive integer)

    Returns:
    The sum of squares of integers from 1 to n, or None if n is invalid.
    """
    if isinstance(n, int) and n > 0:
        return (n * (n + 1) * (2 * n + 1)) // 6
    return None
