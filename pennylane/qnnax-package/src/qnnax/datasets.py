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


def rescale(X: jax.Array, min=0, max=1) -> jax.Array:
    """
    Rescale the data (both axes equally) to be within the correct range
    :param jax.Array X: Input data to scale
    :param float min: New minimum value
    :param float max: New maximum value
    :return: Scaled data
    :rtype: jax.Array
    """
    minx = jnp.min(X)
    X = (X - minx) + min  # linear shift
    X = X / jnp.max(X) * max  # rescale
    return X


def create_circles(seed: int, n=1000, num_features=2, radius=jnp.sqrt(2/jnp.pi), normal=False) -> tuple[jax.Array, jax.Array]:
    """
    Create a dataset of a circle (sphere in higher dimensions) around the origin and if a random point is within or outside the circle.
    :param int seed: Random number generator seed
    :param int n: Number of samples
    :param int num_features: Number of features (dimensions to generate the circle in)
    :param float radius: Radius of the circle (default is 50% odds in 2D)
    :param bool normal: If true, use normally distributed data otherwise uniformly distributed
    :return: Tuple (X, y) of generated data
    :rtype: tuple(jax.Array, jax.Array)
    """
    key = jax.random.PRNGKey(seed)
    if normal:
        X = jax.random.normal(key, shape=(n, num_features), dtype=jnp.float32)
    else:
        X = jax.random.uniform(key, shape=(n, num_features), minval=-1, maxval=1, dtype=jnp.float32)
    y = jnp.where(jnp.sqrt(jnp.sum(X ** 2, axis=1)) > radius, 1, -1)
    return X, y


def create_moons(seed: int, n=100) -> tuple[jax.Array, jax.Array]:
    """
    Create a dataset of interlaced moons in 2D.
    :param int seed: Random number generator seed
    :param int n: Number of samples
    :return: Tuple (X, y) of generated data
    :rtype: tuple(jax.Array, jax.Array)
    """
    X, y = make_moons(n_samples=n, random_state=seed, noise=0.1)
    return jnp.array(X, dtype=jnp.float32), (2 * jnp.array(y)) - 1


def create_blobs(seed: int, n=100) -> tuple[jax.Array, jax.Array]:
    """
    Create a dataset of scattered blobs in 2D.
    :param int seed: Random number generator seed
    :param int n: Number of samples
    :return: Tuple (X, y) of generated data
    :rtype: tuple(jax.Array, jax.Array)
    """
    X, y = make_blobs(n_samples=n, n_features=2, random_state=seed)
    return jnp.array(X, dtype=jnp.float32), (2 * jnp.array(y)) - 1


def create_classification(seed: int, n=100, num_features=2, separation=2.0) -> tuple[jax.Array, jax.Array]:
    """
    Create a dataset of blob classification data. This is highly variable between different seeds.
    :param int seed: Random number generator seed
    :param int n: Number of samples
    :param int num_features: Number of features (dimensions to generate the classification in)
    :param float separation: Separation distance between blobs. Larger values make the problem easier
    :return: Tuple (X, y) of generated data
    :rtype: tuple(jax.Array, jax.Array)
    """
    X, y = make_classification(n_samples=n, n_features=num_features,
                               n_informative=num_features, n_redundant=0, random_state=seed,
                               class_sep=separation,  # how hard the problem should be
                               )
    return jnp.array(X, dtype=jnp.float32), (2 * jnp.array(y)) - 1
