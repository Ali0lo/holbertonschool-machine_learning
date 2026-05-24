#!/usr/bin/env python3
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):

    linkage = scipy.cluster.hierarchy.linkage(
        X,
        method='ward'
    )

    clss = scipy.cluster.hierarchy.fcluster(
        linkage,
        t=dist,
        criterion='distance'
    )

    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    plt.show()

    return clss
