#!/usr/bin/env python3
"""Module for Gaussian Process implementation."""
import numpy as np
class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian Process.
        Args:
            X_init (numpy.ndarray): Shape (t, 1), inputs already sampled
                with the black-box function.
            Y_init (numpy.ndarray): Shape (t, 1), outputs of the black-box
                function for each input in X_init.
            l (float): Length parameter for the kernel.
            sigma_f (float): Standard deviation given to the output of the
                black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)
    def kernel(self, X1, X2):
        """Calculate the covariance kernel matrix between two matrices.
        Uses the Radial Basis Function (RBF) kernel.
        Args:
            X1 (numpy.ndarray): Shape (m, 1).
            X2 (numpy.ndarray): Shape (n, 1).
        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n).
        """
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
            np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
    def predict(self, X_s):
        """Predict the mean and variance of points in the Gaussian process.
        Args:
            X_s (numpy.ndarray): Shape (s, 1), points whose mean and
                standard deviation should be calculated.
        Returns:
            tuple: (mu, sigma)
                - mu (numpy.ndarray): Shape (s,), mean for each point in X_s.
                - sigma (numpy.ndarray): Shape (s,), variance for each point
                  in X_s.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        return mu, sigma
