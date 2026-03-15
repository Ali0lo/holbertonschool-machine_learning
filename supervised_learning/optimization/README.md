# Optimization

## 0. Normalization constants

File: `0-norm_constants.py`

### Function
- `normalization_constants(X)`

### Description
Calculates the feature-wise normalization (standardization) constants for a
dataset matrix `X` of shape `(m, nx)`:
- Mean of each feature (axis `0`)
- Standard deviation of each feature (axis `0`)

### Return
A tuple:
1. `mean` — numpy array of shape `(nx,)`
2. `stddev` — numpy array of shape `(nx,)`
