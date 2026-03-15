#!/usr/bin/env python3
"""Isolation Forest"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """Isolation Random Forest"""
    def __init__(self, n_trees=100, max_depth=10, seed=0):
        """Initialize"""
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees = []

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Fit the isolation forest"""
        self.explanatory = explanatory
        self.trees = []
        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            tree = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i
            )
            tree.fit(explanatory)
            self.trees.append(tree)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))

        if verbose == 1:
            print(
                f"  Training finished.\n"
                f"    - Mean depth                     : {np.mean(depths)}\n"
                f"    - Mean number of nodes           : {np.mean(nodes)}\n"
                f"    - Mean number of leaves          : {np.mean(leaves)}"
            )

    def suspects(self, explanatory, n_suspects=5):
        """Return the n_suspects most likely outliers"""
        all_depths = np.zeros((len(explanatory), len(self.trees)))
        for i, tree in enumerate(self.trees):
            all_depths[:, i] = tree.predict(explanatory)

        mean_depths = all_depths.mean(axis=1)
        idx = np.argsort(mean_depths)[:n_suspects]
        return explanatory[idx], mean_depths[idx]
