#!/usr/bin/env python3
from 3-neural_style import NST
import tensorflow as tf


class NST(NST):

    def layer_style_cost(self, style_output, gram_target):
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        c = style_output.shape[-1]

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape "
                f"[1, {c}, {c}]"
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )
