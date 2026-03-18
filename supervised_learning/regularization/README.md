# L2 Regularized Cost

This project provides a function:

- **`l2_reg_cost(cost, lambtha, weights, L, m)`**

It computes neural network cost with L2 regularization:

\[
J_{reg} = J + \frac{\lambda}{2m}\sum_{l=1}^{L}\|W^{[l]}\|_F^2
\]

## Files

- `0-l2_reg_cost.py` - implementation of L2-regularized cost
- `README.md` - project description

## Requirements

- Python 3
- NumPy

## Example

```python
import numpy as np
from 0-l2_reg_cost import l2_reg_cost
# Note: if import fails due to filename, use importlib to load by path.
```
