#!/usr/bin/env python3
"""PCA Color Augmentation as described in the AlexNet paper."""
import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation on an image.

    This implements the color augmentation from the AlexNet paper:
    fancy PCA over the RGB pixel values of the training set.
    For each image, the principal components of the RGB channel
    covariance matrix are computed, and a multiple of the found
    principal components (scaled by their eigenvalues and random
    alphas) is added to each pixel.

    Args:
        image: a 3D tf.Tensor of shape (H, W, 3) containing the
               image to augment. Expected dtype is uint8 or float.
        alphas: a tuple of length 3 containing the amount that each
                principal color component should change (one per
                principal component direction).

    Returns:
        The augmented image as a tf.Tensor with the same shape as
        the input, clipped to [0, 255] and cast back to uint8.
    """
    img = tf.cast(image, tf.float32)

    pixels = tf.reshape(img, [-1, 3])

    mean = tf.reduce_mean(pixels, axis=0)
    pixels_centered = pixels - mean

    n = tf.cast(tf.shape(pixels_centered)[0], tf.float32)
    cov = tf.matmul(pixels_centered, pixels_centered, transpose_a=True)
    cov = cov / (n - 1.0)

    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    alphas_tensor = tf.cast(alphas, tf.float32)
    scales = alphas_tensor * tf.sqrt(tf.abs(eigenvalues))

    perturbation = tf.linalg.matvec(eigenvectors, scales)

    img_augmented = img + perturbation
    img_augmented = tf.clip_by_value(img_augmented, 0.0, 255.0)

    return tf.cast(img_augmented, tf.uint8)
