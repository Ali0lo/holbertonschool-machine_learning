#!/usr/bin/env python3
import sklearn.mixture


def gmm(X, k):

    model = sklearn.mixture.GaussianMixture(
        n_components=k
    )

    model.fit(X)

    clss = model.predict(X)

    return (
        model.weights_,
        model.means_,
        model.covariances_,
        clss,
        model.bic(X)
    )
