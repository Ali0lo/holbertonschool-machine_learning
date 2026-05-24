#!/usr/bin/env python3
from 1-neural_style import NST
import tensorflow as tf


class NST(NST):

    @staticmethod
    def gram_matrix(input_layer):
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )

        h = input_layer.shape[1]
        w = input_layer.shape[2]

        return result / tf.cast(h * w, tf.float32)
