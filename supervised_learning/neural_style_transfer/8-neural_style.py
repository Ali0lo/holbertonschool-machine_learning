#!/usr/bin/env python3
from 7-neural_style import NST
import tensorflow as tf


class NST(NST):

    def compute_grads(self, generated_image):
        if not isinstance(generated_image,
                          (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = \
                self.total_cost(generated_image)

        grads = tape.gradient(J_total, generated_image)

        return grads, J_total, J_content, J_style
