#!/usr/bin/env python3
""" Adjust brightness of the image """
import tensorflow as tf


def rotate_image(image, lower, upper):
    """ Randomly adjust the brightness """
    return tf.image.random_contrast(image, lower, upper)
