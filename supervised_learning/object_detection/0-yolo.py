#!/usr/bin/env python3
"""
Yolo v3 object detection
"""

import tensorflow.keras as K


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes a Yolo instance

        Args:
            model_path: path to the Darknet Keras model
            classes_path: path to the class names file
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray containing anchor boxes
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
