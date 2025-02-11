# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 10:32 2025

Model plotting functions.

@author: james
"""

from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from qnnax.QNN import QNN


def plot_decision_regions(qnn: QNN, X: jax.Array, y: jax.Array, size=(0.0, 1.0, 0.0, 1.0), nx=40, ny=40, figsize=(7, 5),
                          colors=None, color_map=plt.cm.RdBu):
    """
    Plot the class decision regions for the model overlaid with the training and/or test data.
    :param QNN qnn: Trained QNN model
    :param jax.Array X: Input X data
    :param jax.Array y: Input y data (class labels)
    :param tuple(float, float, float, float) size: Range for the decision region plots. Rescaled to at least be as big/small as the X data. Shape of (minX, maxX, minY, maxY)
    :param int nx: Number of points along X axis to evaluate for the decision region
    :param int ny: Number of points along Y axis to evaluate for the decision region
    :param tuple(int, int) figsize: Matplotlib figure size
    :param list or None colors: Colors to use for plotting. If None uses `['b', 'r']`
    :param color_map: Matplotlib color map for the decision region
    :return: Tuple of Matplotlib figure and axes
    """

    if colors is None:
        colors = ['b', 'r']

    check_is_fitted(qnn)

    # Rescale size to fit data if bigger than the given range
    size = (
        min([size[0], jnp.min(X[:, 0]).item()]),
        max([size[1], jnp.max(X[:, 0]).item()]),
        min([size[2], jnp.min(X[:, 1]).item()]),
        max([size[3], jnp.max(X[:, 1]).item()]),
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # make data for decision regions
    xx, yy = jnp.meshgrid(jnp.linspace(size[0], size[1], nx), jnp.linspace(size[2], size[3], ny))
    X_grid = jnp.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())])
    predictions_grid = qnn.predict(X_grid)
    Z = jnp.reshape(predictions_grid, xx.shape)

    # plot decision regions
    levels = jnp.arange(-1, 1.1, 0.1)
    cnt = ax.contourf(xx, yy, Z, levels=levels, cmap=color_map, alpha=0.8, extend="both")
    ax.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
    fig.colorbar(cnt, ticks=[-1, 0, 1], ax=ax)

    # plot data
    for color, label in zip(colors, [1, -1]):
        plot_x = X[:, 0][y == label]
        plot_y = X[:, 1][y == label]
        ax.scatter(plot_x, plot_y, c=color, marker="o", ec="k", label=f"class {label}")

    return fig, ax


def plot_history(qnn: QNN):
    """
    Plot the training history of the QNN model, including both loss and accuracy.
    :param QNN qnn: Model
    :return: Matplotlib figure and axes
    """
    check_is_fitted(qnn)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(qnn.history['loss'], label="Train")
    axs[1].plot(qnn.history['acc'])

    if qnn.history["test_loss"] is not None:
        axs[0].plot(qnn.history['test_loss'], label="Test")
        axs[1].plot(qnn.history['test_acc'])

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy %')

    return fig, axs

