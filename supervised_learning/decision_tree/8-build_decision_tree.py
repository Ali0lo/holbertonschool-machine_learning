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
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

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

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """ Predict node """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

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

    def pred(self, x):
        """ Predict leaf """
        return self.value


class Decision_Tree:
    """ Class for implementing Decision Tree """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ Initialize Tree """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
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
        return str(self.root)

    def update_predict(self):
        """ Faster predict """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )

    def pred(self, x):
        """ Pred """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """ Fit method """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory,
                                   self.target)}"""
            )

    def np_extrema(self, arr):
        """ Return min and max values """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ Randomly split nodes """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """ fit node """
        node.feature, node.threshold = self.split_criterion(node)

        # individuals that go left (strictly greater than threshold)
        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        # individuals that go right (less than or equal to threshold)
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        # Is left node a leaf?
        is_left_leaf = (
            np.sum(left_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[left_population]).size == 1
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = (
            np.sum(right_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[right_population]).size == 1
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """ Create leaf node """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ Create internal node """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """ accuracy as a metric """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size

    def possible_thresholds(self, node, feature):
        """ Compute possible thresholds """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """ Gini split criterion for one feature """
        thresholds = self.possible_thresholds(node, feature)

        # individuals and classes in this node
        individuals = self.explanatory[:, feature][node.sub_population]
        classes = self.target[node.sub_population]
        unique_classes = np.unique(classes)

        # shape (n, t, c): is individual i > threshold j AND of class k?
        Left_F = (
            (individuals[:, np.newaxis] > thresholds[np.newaxis, :]
             )[:, :, np.newaxis]
            & (classes[:, np.newaxis] == unique_classes[np.newaxis, :]
               )[:, np.newaxis, :]
            )

        # total individuals in node
        n = individuals.shape[0]

        # left child counts per threshold: sum over individuals (axis 0)
        left_counts = Left_F.sum(axis=0)           # shape (t, c)
        left_sizes = left_counts.sum(axis=1)       # shape (t,)

        # right child counts
        total = (classes == unique_classes[:, np.newaxis]).T.sum(axis=0)
        right_counts = total - left_counts
        right_sizes = n - left_sizes               # shape (t,)

        # Gini = 1 - sum of squared proportions
        # avoid division by zero
        left_sizes_safe = np.where(left_sizes == 0, 1, left_sizes)
        right_sizes_safe = np.where(right_sizes == 0, 1, right_sizes)

        left_gini = 1 - np.sum(
            (left_counts / left_sizes_safe[:, np.newaxis]) ** 2, axis=1
        )
        right_gini = 1 - np.sum(
            (right_counts / right_sizes_safe[:, np.newaxis]) ** 2, axis=1
        )

        # weighted average Gini
        gini_split = (left_sizes * left_gini + right_sizes * right_gini) / n

        # return best threshold and its gini score
        best = np.argmin(gini_split)
        return thresholds[best], gini_split[best]

    def Gini_split_criterion(self, node):
        """ Gini split criterion """
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
