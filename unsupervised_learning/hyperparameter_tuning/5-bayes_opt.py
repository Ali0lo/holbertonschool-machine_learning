#!/usr/bin/env python3
"""Module for Bayesian Optimization implementation."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess
class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initialize Bayesian Optimization.
        Args:
            f (callable): The black-box function to be optimized.
            X_init (numpy.ndarray): Shape (t, 1), inputs already sampled
                with the black-box function.
            Y_init (numpy.ndarray): Shape (t, 1), outputs of the black-box
                function for each input in X_init.
            bounds (tuple): (min, max) representing the bounds of the space
                in which to look for the optimal point.
            ac_samples (int): Number of samples to analyze during acquisition.
            l (float): Length parameter for the kernel.
            sigma_f (float): Standard deviation given to the output of the
                black-box function.
            xsi (float): Exploration-exploitation factor for acquisition.
            minimize (bool): True for minimization, False for maximization.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
    def acquisition(self):
        """Calculate the next best sample location using Expected Improvement.
        Returns:
            tuple: (X_next, EI)
                - X_next (numpy.ndarray): Shape (1,), the next best sample
                  point.
                - EI (numpy.ndarray): Shape (ac_samples,), the expected
                  improvement of each potential sample.
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            Y_opt = np.min(self.gp.Y)
            imp = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            imp = mu - Y_opt - self.xsi
        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(EI)].reshape(-1)
        return X_next, EI
    def optimize(self, iterations=100):
        """Optimize the black-box function using Bayesian optimization.
        Stops early if the next proposed point has already been sampled.
        Args:
            iterations (int): Maximum number of iterations to perform.
        Returns:
            tuple: (X_opt, Y_opt)
                - X_opt (numpy.ndarray): Shape (1,), the optimal point.
                - Y_opt (numpy.ndarray): Shape (1,), the optimal function
                  value.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if np.any(np.isclose(self.gp.X, X_next)):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx].reshape(-1)
        Y_opt = self.gp.Y[idx].reshape(-1)
        return X_opt, Y_opt
