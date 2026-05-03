#!/usr/bin/env python3
"""
Yolo v3 object detection
"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes a Yolo instance
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs.

        Args:
            outputs: list of numpy.ndarrays containing predictions
            image_size: numpy.ndarray containing original image size
                        as [image_height, image_width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        image_height = image_size[0]
        image_width = image_size[1]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_width
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_height

            anchor_width = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes)
            anchor_height = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes)

            bw = (np.exp(tw) * anchor_width) / input_width
            bh = (np.exp(th) * anchor_height) / input_height

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on their objectness score and class probability.

        Args:
            boxes: list of processed boundary boxes
            box_confidences: list of box confidence scores
            box_class_probs: list of box class probabilities

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(
            boxes, box_confidences, box_class_probs
        ):
            scores = confidence * class_probs

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filtered boxes.

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4)
            box_classes: numpy.ndarray of shape (?,)
            box_scores: numpy.ndarray of shape (?,)

        Returns:
            tuple: (box_predictions, predicted_box_classes,
                    predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]
            sorted_idxs = np.argsort(cls_scores)[::-1]

            while sorted_idxs.size > 0:
                best_idx = sorted_idxs[0]

                box_predictions.append(cls_boxes[best_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[best_idx])

                if sorted_idxs.size == 1:
                    break

                current_box = cls_boxes[best_idx]
                remaining_boxes = cls_boxes[sorted_idxs[1:]]

                x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
                y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
                x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
                y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

                inter_width = np.maximum(0, x2 - x1)
                inter_height = np.maximum(0, y2 - y1)
                intersection = inter_width * inter_height

                current_area = (
                    (current_box[2] - current_box[0]) *
                    (current_box[3] - current_box[1])
                )
                remaining_areas = (
                    (remaining_boxes[:, 2] - remaining_boxes[:, 0]) *
                    (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                )

                union = current_area + remaining_areas - intersection
                iou = intersection / union

                sorted_idxs = sorted_idxs[1:][iou <= self.nms_t]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
