#!/usr/bin/env python3
""" Change the hue of the image """
import tensorflow as tf


def change_hue(image, delta):
    """ Randomly change the brightness """
    return tf.image.random_brightness(image, max_delta)
