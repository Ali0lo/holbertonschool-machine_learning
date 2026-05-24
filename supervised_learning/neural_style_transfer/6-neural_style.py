#!/usr/bin/env python3
from 5-neural_style import NST
import tensorflow as tf


class NST(NST):

    def content_cost(self, content_output):
        if not isinstance(content_output,
                          (tf.Tensor, tf.Variable)) or \
           content_output.shape != self.content_feature.shape:
            raise TypeError(
                f"content_output must be a tensor of shape "
                f"{self.content_feature.shape}"
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )
