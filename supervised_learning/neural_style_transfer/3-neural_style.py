#!/usr/bin/env python3
from 2-neural_style import NST
import tensorflow as tf


class NST(NST):

    def generate_features(self):
        style_inputs = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )

        content_inputs = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_inputs)
        content_outputs = self.model(content_inputs)

        style_features = style_outputs[:-1]
        content_feature = content_outputs[-1]

        self.gram_style_features = [
            self.gram_matrix(feature)
            for feature in style_features
        ]

        self.content_feature = content_feature

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        super().__init__(style_image, content_image,
                         alpha, beta)

        self.generate_features()
