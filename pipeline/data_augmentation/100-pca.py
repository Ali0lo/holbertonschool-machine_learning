#!/usr/bin/env python3
"""PCA Color Augmentation"""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation on an image"""
    image = tf.cast(image, tf.float32)

    pixels = tf.reshape(image, (-1, 3))
    mean = tf.reduce_mean(pixels, axis=0)
    centered = pixels - mean

    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(centered)[0] - 1, tf.float32)

    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    eigenvalues = tf.reverse(eigenvalues, axis=[0])
    eigenvectors = tf.reverse(eigenvectors, axis=[1])

    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(
        eigenvectors,
        tf.reshape(alphas * eigenvalues, (3, 1))
    )

    image = image + tf.reshape(delta, (1, 1, 3))
    image = tf.clip_by_value(image, 0, 255)

    return tf.cast(image, tf.uint8)