# Keras - Sequential Model

Implements `build_model(nx, layers, activations, lambtha, keep_prob)` using Keras Sequential API.

- Dense layers with L2 regularization (`lambtha`)
- Dropout after hidden layers with rate `1 - keep_prob`
- No use of `Input` class
