#!/usr/bin/env python3
"""Decision tree classes."""


class Node:
    """Node class for a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.depth = depth
        self.is_leaf = False

    def left_child_add_prefix(self, text):
        """Add prefix for left child subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child subtree."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """Return string representation of node and subtree."""
        if self.is_root:
            text = "root [feature={}, threshold={}]".format(
                self.feature, self.threshold
            )
        else:
            text = "node [feature={}, threshold={}]".format(
                self.feature, self.threshold
            )
        text += "\n"
        text += self.left_child_add_prefix(str(self.left_child))
        text += self.right_child_add_prefix(str(self.right_child))
        return text[:-1]


class Leaf:
    """Leaf class for a decision tree."""

    def __init__(self, value, depth=0):
        """Initialize a leaf."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Return string representation of leaf."""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree:
    """Decision tree wrapper."""

    def __init__(self, root=None):
        """Initialize tree."""
        self.root = root

    def __str__(self):
        """Return string representation of the tree."""
        return self.root.__str__()
