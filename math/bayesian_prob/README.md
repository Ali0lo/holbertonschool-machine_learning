# Bayesian Probability

This project implements probability distributions and Bayesian inference calculations.

## Files

### Probability Distributions
- **poisson.py** - Poisson distribution (PMF, CDF)
- **exponential.py** - Exponential distribution (PDF, CDF)
- **normal.py** - Normal distribution (z-score, PDF, CDF)
- **binomial.py** - Binomial distribution (PMF, CDF)

### Bayesian Inference
- **0-likelihood.py** - Calculates likelihood P(X | P)
- **1-intersection.py** - Calculates intersection P(X ∩ P)
- **2-marginal.py** - Calculates marginal probability P(X)
- **3-posterior.py** - Calculates posterior probability P(P | X)

## Bayes' Theorem

**P(A | B) = P(B | A) × P(A) / P(B)**

- **P(A | B)** - Posterior probability
- **P(B | A)** - Likelihood
- **P(A)** - Prior probability
- **P(B)** - Marginal probability

## Requirements
- Python 3.x
- NumPy

## Author
Machine Learning - Probability Project
