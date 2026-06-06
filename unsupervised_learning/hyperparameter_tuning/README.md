# Hyperparameter Tuning

## Files

- **0-gp.py** — `GaussianProcess` class with `__init__` and `kernel` (RBF) methods.
- **1-gp.py** — Adds `predict` method: returns mean and variance for new input points.
- **2-gp.py** — Adds `update` method: incorporates a new sample into the GP.
- **3-bayes_opt.py** — `BayesianOptimization` class with `__init__`; sets up GP and sample space.
- **4-bayes_opt.py** — Adds `acquisition` method: Expected Improvement (EI) for next sample point.
- **5-bayes_opt.py** — Adds `optimize` method: runs full Bayesian optimization loop with early stopping.
