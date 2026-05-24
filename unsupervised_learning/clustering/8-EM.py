#!/usr/bin/env python3
"""Expectation Maximization for GMM"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(
        X, k, iterations=1000,
        tol=1e-5, verbose=False):
    """performs expectation maximization"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None:
        return None, None, None, None, None

    g, l_old = expectation(X, pi, m, S)

    if verbose:
        print(
            "Log Likelihood after 0 iterations: {:.5f}".format(
                l_old
            )
        )

    for i in range(1, iterations + 1):

        pi, m, S = maximization(X, g)

        if pi is None:
            return None, None, None, None, None

        g, l_new = expectation(X, pi, m, S)

        if verbose and (i % 10 == 0 or i == iterations):
            print(
                "Log Likelihood after {} iterations: {:.5f}".format(
                    i, l_new
                )
            )

        if abs(l_new - l_old) <= tol:

            if verbose and i % 10 != 0:
                print(
                    "Log Likelihood after {} iterations: {:.5f}".format(
                        i, l_new
                    )
                )

            return pi, m, S, g, l_new

        l_old = l_new

    return pi, m, S, g, l_new
