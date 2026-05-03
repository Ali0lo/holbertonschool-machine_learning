# YOLO Object Detection

This project implements parts of the YOLO v3 object detection algorithm using TensorFlow/Keras and NumPy.

## Files

- `0-yolo.py`: Initializes the `Yolo` class.
- `1-yolo.py`: Adds output processing for Darknet predictions.

## Class

### Yolo

The `Yolo` class loads a Darknet Keras model, class names, score thresholds, non-max suppression thresholds, and anchor boxes.

## Methods

### `__init__(self, model_path, classes_path, class_t, nms_t, anchors)`

Initializes the YOLO model.

### `process_outputs(self, outputs, image_size)`

Processes the raw outputs from the Darknet model.

Returns:

- `boxes`: processed bounding boxes relative to the original image
- `box_confidences`: confidence scores for each box
- `box_class_probs`: class probabilities for each box
