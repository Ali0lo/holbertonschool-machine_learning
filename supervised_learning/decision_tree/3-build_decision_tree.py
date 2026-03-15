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
            text = "-> node [feature={}, threshold={}]".format(
                self.feature, self.threshold
            )
        text += "\n"
        text += self.left_child_add_prefix(str(self.left_child))
        text += self.right_child_add_prefix(str(self.right_child))
        return text[:-1]

    def max_depth_below(self):
        """Return maximum depth below this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node."""
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def get_leaves_below(self):
        """Return list of all leaves below this node."""
        return (
            self.left_child.get_leaves_below() +
            self.right_child.get_leaves_below()
        )


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

    def max_depth_below(self):
        """Return depth for a leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below leaf."""
        return 1

    def get_leaves_below(self):
        """Return this leaf in a list."""
        return [self]


class Decision_Tree:
    """Decision tree wrapper."""

    def __init__(self, root=None):
        """Initialize tree."""
        self.root = root

    def __str__(self):
        """Return string representation of the tree."""
        return self.root.__str__()

    def depth(self):
        """Return maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Return list of all leaves in the tree."""
        return self.root.get_leaves_below()
