#!/usr/bin/env python3
""" Adjust contarst of the image """
import tensorflow as tf


def change_contrast(image, lower, upper):
    """ Randomly adjust the contrast """
    return tf.image.random_contrast(image, lower, upper)
