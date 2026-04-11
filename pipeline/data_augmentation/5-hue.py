#!/usr/bin/env python3
""" Change the hue of the image """
import tensorflow as tf


def change_hue(image, delta):
    """ Change hue """
    return tf.image.adjust_hue(image, delta)
