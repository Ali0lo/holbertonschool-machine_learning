#!/usr/bin/env python3
"""
Yolo v3 object detection
"""

import os
import cv2
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
        Processes Darknet model outputs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h = image_size[0]
        image_w = image_size[1]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)

            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_w
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_h

            anchor_w = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes)

            bw = (np.exp(tw) * anchor_w) / input_w
            bh = (np.exp(th) * anchor_h) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.zeros((grid_h, grid_w, anchor_boxes, 4))
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
        Filters boxes based on box scores
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

        return (
            np.concatenate(filtered_boxes, axis=0),
            np.concatenate(box_classes, axis=0),
            np.concatenate(box_scores, axis=0),
        )

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idxs = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]
            order = np.argsort(cls_scores)[::-1]

            while order.size > 0:
                best = order[0]

                box_predictions.append(cls_boxes[best])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[best])

                if order.size == 1:
                    break

                current = cls_boxes[best]
                rest = cls_boxes[order[1:]]

                x1 = np.maximum(current[0], rest[:, 0])
                y1 = np.maximum(current[1], rest[:, 1])
                x2 = np.minimum(current[2], rest[:, 2])
                y2 = np.minimum(current[3], rest[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                current_area = (
                    (current[2] - current[0]) *
                    (current[3] - current[1])
                )
                rest_area = (
                    (rest[:, 2] - rest[:, 0]) *
                    (rest[:, 3] - rest[:, 1])
                )

                union = current_area + rest_area - inter_area
                iou = inter_area / union

                order = order[1:][iou <= self.nms_t]

        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores),
        )

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)

            if image is not None:
                images.append(image)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for the Darknet model
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            pimages.append(resized / 255)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays an image with predicted boxes
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            text = "{} {:.2f}".format(class_name, score)

            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord("s"):
            if not os.path.exists("detections"):
                os.makedirs("detections")

            save_path = os.path.join("detections", file_name)
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Makes predictions on all images in a folder
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []

        for i, image in enumerate(images):
            output = [outputs[j][i] for j in range(len(outputs))]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                output,
                image_shapes[i]
            )

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes,
                box_confidences,
                box_class_probs
            )

            boxes, box_classes, box_scores = self.non_max_suppression(
                filtered_boxes,
                box_classes,
                box_scores
            )

            predictions.append((boxes, box_classes, box_scores))

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(
                image.copy(),
                boxes,
                box_classes,
                box_scores,
                file_name
            )

        return predictions, image_paths
