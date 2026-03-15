#!/usr/bin/env python3
""" Implementing Random Forest """
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """ Random Forest """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """ Initialize """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """ Predict using majority vote across all trees """
        predictions = np.array([pred(explanatory)
                               for pred in self.numpy_preds])
        # predictions shape: (n_trees, n_individuals)
        # for each individual, pick the most frequent class
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """ fit method """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(
                f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
                                   self.target)}"""
            )

    def accuracy(self, test_explanatory, test_target):
        """ accuracy metric """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
