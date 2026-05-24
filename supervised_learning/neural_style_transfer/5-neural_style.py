#!/usr/bin/env python3
from 4-neural_style import NST
import tensorflow as tf


class NST(NST):

    def style_cost(self, style_outputs):
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != len(self.style_layers):
            raise TypeError(
                f"style_outputs must be a list with a length of "
                f"{len(self.style_layers)}"
            )

        weight = 1 / len(self.style_layers)

        cost = 0

        for output, target in zip(
                style_outputs,
                self.gram_style_features):
            cost += weight * self.layer_style_cost(
                output,
                target
            )

        return cost
