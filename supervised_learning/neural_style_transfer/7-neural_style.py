#!/usr/bin/env python3
from 6-neural_style import NST
import tensorflow as tf


class NST(NST):

    def total_cost(self, generated_image):
        if not isinstance(generated_image,
                          (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        inputs = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )

        outputs = self.model(inputs)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)

        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style
