#!/usr/bin/env python3
import sklearn.cluster


def kmeans(X, k):

    model = sklearn.cluster.KMeans(n_clusters=k)

    clss = model.fit_predict(X)

    return model.cluster_centers_, clss
