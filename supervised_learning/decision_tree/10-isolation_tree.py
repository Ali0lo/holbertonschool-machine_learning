#!/usr/bin/env python3
""" Isolation Forest Part I """
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """ isolation forest """
    def __init__(self, max_depth=10, seed=0, root=None):
        """ initalize class """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """ print tree """
        return str(self.root)

    def depth(self):
        """ calculate depth """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ count nodes """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """ update bounds """
        self.root.update_bounds_below()

    def get_leaves(self):
        """ get leaves """
        return self.root.get_leaves_below()

    def update_predict(self):
        """Return depth of the leaf each individual falls into"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.depth for leaf in leaves]),
            axis=0
        )

    def np_extrema(self, arr):
        """Return min and max values"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ random spliting criteria """
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

    def get_leaf_child(self, node, sub_population):
        """Leaf stores its depth as value"""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ get node child """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """ fit method """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (
            node.sub_population
            & (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = (
            node.sub_population
            & (self.explanatory[:, node.feature] <= node.threshold)
        )

        # Leaf if only 1 individual or max depth reached
        is_left_leaf = (
            np.sum(left_population) <= 1 or node.depth + 1 >= self.max_depth
        )
        is_right_leaf = (
            np.sum(right_population) <= 1 or node.depth + 1 >= self.max_depth
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """ fit """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}"""
            )
