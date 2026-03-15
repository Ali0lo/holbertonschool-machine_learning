# Classification Metrics

## 0. Create Confusion Matrix
`0-create_confusion.py` contains:

- `create_confusion_matrix(labels, logits)`
  - Builds a confusion matrix from one-hot true labels and one-hot predicted labels.
  - Rows represent actual classes.
  - Columns represent predicted classes.

## 1. Sensitivity
`1-sensitivity.py` contains:

- `sensitivity(confusion)`
  - Computes sensitivity (recall) for each class:
    - `TP / (TP + FN)`

