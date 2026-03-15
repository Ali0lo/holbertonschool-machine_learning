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
        self.lower = {}  # lower bounds per feature
        self.upper = {}  # upper bounds per feature

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
        if self.is_leaf:
            return [self]

        leaves = []

        if self.left_child:
            leaves += self.left_child.get_leaves_below()

        if self.right_child:
            leaves += self.right_child.get_leaves_below()

        return leaves

    def update_bounds_below(self):
        """ Update bounds below function """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            # Copy parent bounds
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            # Update bounds for the splitting feature
            if self.feature is not None:
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        # Recurse
        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """ update indicator """
        def is_large_enough(x):
            """ check if feature satisfies lower bound """
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                         for key in self.lower.keys()]),
                axis=0
            )

        def is_small_enough(x):
            """ check if feature satisfies upper bound """
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                         for key in self.upper.keys()]),
                axis=0
            )

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]
            ), axis=0)

    def pred(self, x):
        """ Predict node """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    def __str__(self):
        """ Print node for debugging """
        s = f"Node(feature={self.feature}, threshold={self.threshold})"
        return s


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

    def pred(self, x):
        """ Predict leaf """
        return self.value


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

    def update_predict(self):
        """ Faster predict """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(np.array([
            leaf.indicator(A) * leaf.value
            for leaf in leaves
            ]), axis=0)

    def pred(self, x):
        """ Pred """
        return self.root.pred(x)
