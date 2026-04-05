# Convolution Forward Propagation

This project contains `0-conv_forward.py` with:

- `conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1))`

## Description

The function performs forward propagation for a convolutional neural network layer using NumPy.

### Inputs

- `A_prev`: `(m, h_prev, w_prev, c_prev)`
- `W`: `(kh, kw, c_prev, c_new)`
- `b`: `(1, 1, 1, c_new)`
- `activation`: activation function
- `padding`: `"same"` or `"valid"`
- `stride`: `(sh, sw)`

### Output

- Activated convolution output as a NumPy array.

## Notes

- `"same"` padding keeps spatial size approximately aligned with stride.
- `"valid"` padding applies no zero-padding.
