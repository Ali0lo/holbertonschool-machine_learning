#!/usr/bin/env python3
"""PCA Color Augmentation as described in the AlexNet paper."""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation on an image.

    Args:
        image: 3D tf.Tensor containing the image to change
        alphas: tuple of length 3 containing the amount each
                principal component should change

    Returns:
        The augmented image
    """
    image = tf.cast(image, tf.float32)

    pixels = tf.reshape(image, (-1, 3))
    mean = tf.reduce_mean(pixels, axis=0)
    centered = pixels - mean

    n = tf.cast(tf.shape(centered)[0], tf.float32)
    cov = tf.matmul(centered, centered, transpose_a=True) / (n - 1)

    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(
        eigenvectors,
        tf.reshape(alphas * eigenvalues, (3, 1))
    )
    delta = tf.reshape(delta, (1, 1, 3))

    augmented = image + delta
    augmented = tf.clip_by_value(augmented, 0, 255)

    return tf.cast(augmented, tf.uint8)