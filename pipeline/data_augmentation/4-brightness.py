#!/usr/bin/env python3
""" Adjust brightness of the image """
import tensorflow as tf


def change_brightness(image, max_delta):
    """ Randomly change the brightness """
    return tf.image.random_brightness(image, max_delta)
