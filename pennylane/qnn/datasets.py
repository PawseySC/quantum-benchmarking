# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:10 2025

Various datasets to test the models against.

All return a binary classification problem with labels of {-1, 1}

@author: james
"""


import jax
import jax.numpy as jnp
from sklearn.datasets import make_classification, make_blobs, make_moons


def rescale(X, min=0, max=1):
    """
    Rescale the data (both axes equally) to be within the correct range
    :param X: Data
    :param min: New minimum x value
    :param max:  New maximum x value
    :return: New data
    """
    minx = jnp.min(X)
    X = (X - minx) + min  # linear shift
    X = X / jnp.max(X) * max  # rescale
    return X


def create_circles(seed, n=1000, num_features=2, radius=jnp.sqrt(2/jnp.pi)):
    # Radius chosen to give 50/50 prob of equal area
    key = jax.random.PRNGKey(seed)
    #X = jax.random.normal(key, shape=(n, num_features))
    X = jax.random.uniform(key, shape=(n, num_features), minval=-1, maxval=1, dtype=jnp.float32)
    y = jnp.where(jnp.sqrt(jnp.sum(X ** 2, axis=1)) > radius, 1, -1)
    return X, y


def create_moons(seed, n=100):
    X, y = make_moons(n_samples=n, random_state=seed, noise=0.1)
    return jnp.array(X, dtype=jnp.float32), (2 * jnp.array(y)) - 1


def create_blobs(seed, n=100):
    X, y = make_blobs(n_samples=n, n_features=2, random_state=seed)
    return jnp.array(X, dtype=jnp.float32), (2 * jnp.array(y)) - 1


def create_classification(seed, n=100, num_features=2, seperation=2.0):
    X, y = make_classification(n_samples=n, n_features=num_features,
                               n_informative=num_features, n_redundant=0, random_state=seed,
                               class_sep=seperation,  # how hard the problem should be
                               )
    return jnp.array(X, dtype=jnp.float32), (2 * jnp.array(y)) - 1
