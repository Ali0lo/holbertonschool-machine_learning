#!/usr/bin/env python3
from 9-neural_style import NST
import tensorflow as tf
import numpy as np


class NST(NST):

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1, var=10):

        if not isinstance(var, (float, int)) or var < 0:
            raise TypeError(
                "var must be a non-negative number"
            )

        self.var = var

        super().__init__(
            style_image,
            content_image,
            alpha,
            beta
        )

    @staticmethod
    def variational_cost(generated_image):
        return tf.image.total_variation(
            generated_image
        )

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
        J_var = self.variational_cost(generated_image)

        J = self.alpha * J_content + \
            self.beta * J_style + \
            self.var * J_var

        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        with tf.GradientTape() as tape:
            J_total, J_content, J_style, J_var = \
                self.total_cost(generated_image)

        grads = tape.gradient(J_total, generated_image)

        return grads, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000,
                       step=None,
                       lr=0.01,
                       beta1=0.9,
                       beta2=0.99):

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):

            grads, J_total, J_content, \
                J_style, J_var = \
                self.compute_grads(generated_image)

            optimizer.apply_gradients(
                [(grads, generated_image)]
            )

            generated_image.assign(
                tf.clip_by_value(generated_image, 0, 1)
            )

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and \
               (i % step == 0 or i == iterations):

                print(
                    f"Cost at iteration {i}: "
                    f"{J_total}, "
                    f"content {J_content}, "
                    f"style {J_style}, "
                    f"var {J_var}"
                )

        return best_image, best_cost
