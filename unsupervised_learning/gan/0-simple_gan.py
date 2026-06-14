#!/usr/bin/env python3
"""Module defines the Simple_GAN class."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """Simple Generative Adversarial Network.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        latent_generator: Function generating latent vectors.
        real_examples: Dataset of real examples.
        batch_size: Batch size.
        disc_iter: Number of discriminator iterations.
        learning_rate: Learning rate.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the GAN."""
        ...
