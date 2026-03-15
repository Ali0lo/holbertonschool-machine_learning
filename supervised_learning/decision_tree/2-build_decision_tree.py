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
        """ Return all leaves below node """
        return (
            self.left_child.get_leaves_below() +
            self.right_child.get_leaves_below()
        )

    def left_child_add_prefix(self, text):
        """ Add prefix to left child """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ Add prefix to right child """
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """ Print node """
        if self.is_root:
            label = (
                f"root [feature={self.feature}, threshold={self.threshold}]"
            )
        else:
            label = (
                f"-> node [feature={self.feature}, threshold={self.threshold}]"
            )
        left_str = self.left_child_add_prefix(self.left_child.__str__())
        right_str = self.right_child_add_prefix(self.right_child.__str__())
        return label + "\n" + left_str + right_str[:-1]


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
        """ Return this leaf """
        return [self]

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
        """ Return leaves """
        return self.root.get_leaves_below()

    def __str__(self):
        """ Print tree """
        return self.root.__str__()
