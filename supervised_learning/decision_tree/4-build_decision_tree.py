#!/usr/bin/env python3
""" Implementing Decision Tree """
import numpy as np


class Node:
    """ The node of the tree """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """ Initialize Node """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        """ Function for finding depth of the tree """
        if self.is_leaf:
            return self.depth
        return max(
            self.depth,
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """ Count nodes """
        if self.is_leaf:
            return 1
        right_count = self.right_child.count_nodes_below(only_leaves)
        left_count = self.left_child.count_nodes_below(only_leaves)
        if only_leaves:
            return right_count + left_count
        return 1 + right_count + left_count

    def get_leaves_below(self):
        """ Return list of all leaves below this node """
        return (
            self.left_child.get_leaves_below() +
            self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """ Update bounds below function """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            if child is self.left_child:
                child.upper[self.feature] = self.threshold
            else:
                child.lower[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def __str__(self):
        """ Print node for debugging """
        return f"Node(feature={self.feature}, threshold={self.threshold})"


class Leaf(Node):
    """ The leaf of the tree """
    def __init__(self, value, depth=None):
        """ Initialize leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Depth of the tree """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Count nodes """
        return 1

    def get_leaves_below(self):
        """ return current leaf """
        return [self]

    def update_bounds_below(self):
        """ Leaf does not propagate bounds """
        pass

    def __str__(self):
        """ Print the leaf of the tree """
        return f"-> leaf [value={self.value}]"


class Decision_Tree():
    """ Class for implementing Decision Tree """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ Initialize Tree """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ Function for finding depth of the tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Count nodes """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """ list of all leaves """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Update lower and upper bounds for the entire tree """
        self.root.update_bounds_below()

    def __str__(self):
        """ Print tree """
        return self.root.__str__()
