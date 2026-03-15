#!/usr/bin/env python3
"""Decision tree classes."""


class Node:
    """Node class for a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Class constructor."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.depth = depth
        self.is_leaf = False

    def max_depth_below(self):
        """Returns the maximum depth in the subtree rooted at this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes in the subtree rooted at this node."""
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count


class Leaf:
    """Leaf class for a decision tree."""

    def __init__(self, value, depth=0):
        """Class constructor."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def max_depth_below(self):
        """Returns leaf depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes below leaf (always 1)."""
        return 1


class Decision_Tree:
    """Decision tree wrapper."""

    def __init__(self, root=None):
        """Class constructor."""
        self.root = root

    def depth(self):
        """Returns the maximum depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)
