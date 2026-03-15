# Deep Learning Optimization Techniques (Quick Guide)

![Optimization Techniques Diagram](https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?auto=format&fit=crop&w=1400&q=80)

## Feature Scaling
Feature scaling standardizes input features to similar ranges (for example, zero mean and unit variance), preventing large-scale features from dominating gradient updates. It usually makes training faster and more stable.

## Batch Normalization
Batch normalization normalizes activations within each mini-batch and then applies learnable scale (`gamma`) and shift (`beta`) parameters. It often accelerates convergence and improves training stability, although very small batch sizes can introduce noisy statistics.

## Mini-batch Gradient Descent
Mini-batch gradient descent updates model parameters using small subsets of the dataset rather than all samples at once. This provides a practical balance between computational efficiency and stable gradient estimates.

## Gradient Descent with Momentum
Momentum keeps a running average of previous gradients and uses it to smooth updates in consistent directions. This helps reduce oscillation and usually converges faster than plain gradient descent.

## RMSProp Optimization
RMSProp adapts the learning rate per parameter by dividing gradients by a moving average of squared gradients. It is especially useful when gradients vary in scale or are noisy.

## Adam Optimization
Adam combines momentum (first moment) with RMSProp-like adaptive scaling (second moment), including bias correction for both. It is widely used as a strong default optimizer because it tends to converge quickly across many problems.

## Learning Rate Decay
Learning rate decay reduces the learning rate over time so training can make large early progress and finer late-stage updates. This often improves final convergence and validation performance.
