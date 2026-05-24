#!/usr/bin/env python3
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(
        X, k, iterations=1000,
        tol=1e-5, verbose=False):

    pi, m, S = initialize(X, k)

    g, l_old = expectation(X, pi, m, S)

    for i in range(iterations):

        pi, m, S = maximization(X, g)

        g, l_new = expectation(X, pi, m, S)

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(
                "Log Likelihood after {} iterations: {:.5f}".format(
                    i, l_new
                )
            )

        if abs(l_new - l_old) <= tol:
            break

        l_old = l_new

    return pi, m, S, g, l_new
