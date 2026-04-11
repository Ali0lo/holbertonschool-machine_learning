#!/usr/bin/env python3
""" Adjust contarst of the image """
import tensorflow as tf


def rotate_image(image, lower, upper):
    """ Randomly adjust the contrast """
    return tf.image.random_contrast(image, lower, upper)
