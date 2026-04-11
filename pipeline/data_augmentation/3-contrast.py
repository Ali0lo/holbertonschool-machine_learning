#!/usr/bin/env python3
""" Adjust contarst of the image """
import tensorflow as tf


def rotate_image(image, lower, upper):
    """ Randomly adjust the contrast """
    image = tf.cast(image, tf.float32)
    return tf.image.random_contrast(image, lower, upper)
