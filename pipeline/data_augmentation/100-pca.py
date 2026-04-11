#!/usr/bin/env python3
import tensorflow as tf
""" PCA color augmentation """


def pca_color(image, alphas):
    """
    Performs PCA color augmentation
    """
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)
    flat = tf.reshape(image, [-1, 3])
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flat)[0], tf.float32)
    eigvals, eigvecs = tf.linalg.eigh(cov)
    alphas = tf.constant(alphas, dtype=tf.float32)
    delta = tf.matmul(eigvecs, tf.expand_dims(eigvals * alphas, axis=1))
    delta = tf.squeeze(delta)
    augmented = flat + delta
    augmented = tf.reshape(augmented, shape)
    augmented = tf.clip_by_value(augmented, 0, 255)
    return tf.cast(augmented, tf.uint8)
