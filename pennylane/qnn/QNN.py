# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 08:27 2025

Base class for a QNN model

@author: james
"""

from abc import abstractmethod

from mpi4py import MPI
import mpi4jax

import jax
import jax.numpy as jnp
import pennylane as qml
import optax

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted


def accuracy(labels, predictions):
    return jnp.mean(jnp.equal(labels, predictions))


class QNN(ClassifierMixin, BaseEstimator):
    @abstractmethod
    def __init__(self,
                 dev_type="default.qubit",
                 random_state=37,
                 batch_size=32,
                 epochs=30,
                 learning_rate=0.01,
                 optimizer=optax.adam,
                 comm=None,  # Pass MPI.COMM_WORLD if wishing to use MPI
                 root_proc=0):
        # Make sure that all variables are explicitly set here
        self.dev_type = dev_type
        self.random_state = random_state
        self.key = jax.random.PRNGKey(random_state)  # create the random key
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        if comm is not None:
            # We should use MPI
            self.comm = comm
            self.use_mpi = True
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.root_proc = root_proc
            self.is_root_proc = self.use_mpi and (self.rank == self.root_proc)  # if we are the root process
            self.optimizer = optax.MultiSteps(optimizer(self.learning_rate), every_k_schedule=self.size)  # process every self.size updates to keep in sync
            #self._log(f"Using MPI. Is root process = {self.is_root_proc}")
        else:
            self.comm = None
            self.use_mpi = False
            self.is_root_proc = False
            self.size = 1  # useful for control flow
            self.optimizer = optimizer(self.learning_rate)

        # We will set these later in self.initialise() and self.create_circuit()
        self.params = None  # Will be a dictionary
        self.circuit = None
        self.forward = None  # vmap version of the jitted circuit
        self.history = None  # history of training loss and accuracy
        self._is_fitted = False

    @abstractmethod
    def initialise(self):
        """
        Initialise the weights and bias with random values
        """
        self.params = {}  # create weights here
        self.create_circuit()

    @abstractmethod
    def create_circuit(self):
        """
        Create the Pennylane variational circuit
        :return: vmap version of the jitted circuit
        """
        self.circuit = None
        def forward_fn(params, x):
            # return circuit(params["weights"], x) + params["bias"]
            pass
        self.forward = jax.vmap(jax.jit(forward_fn), in_axes=(None, 0))
        return self.forward

    @abstractmethod
    def transform(self, X):
        """
        Transform the input to valid 4D
        :param X:  Data of shape (n_samples, 2)
        :return:  Data of shape (n_samples, 4)
        """
        return X

    def _log(self, message, silence=False):
        # Log a message with indicator of the rank
        if not silence:
            if self.use_mpi:
                print(f"[{self.rank}] {message}")
            else:
                print(message)

    def fit(self, X, y, X_test=None, y_test=None, silence=False):
        do_validation = (X_test is not None) and (y_test is not None)
        if do_validation:
            X_test = self.transform(X_test)

        X, y = validate_data(self, X, y)  # checks shape
        X = self.transform(X)

        self._log("Compiling circuits...", silence)

        self.initialise()  # define weights and bias and circuit

        opt_state = self.optimizer.init(self.params)

        def cost_fn(params, X, Y):
            preds = self.forward(params, X)
            # Calculate the square loss
            return jnp.mean((Y - qml.math.stack(preds)) ** 2)

        grad_cost_fn = jax.jit(jax.grad(cost_fn))
        num_X = X.shape[0]

        loss_history = []
        acc_history = []
        test_acc_history = []
        test_loss_history = []

        self._log(f"Training with {self.size * self.batch_size} samples per epoch", silence)

        # Training loop
        for epoch in range(self.epochs):
            if self.use_mpi:
                # Calculate the grads in a distributed way

                # note that only self.root_proc will use these indices. the others are overwritten (i think the performance is negligible)
                batch_inxs = jax.random.randint(key=self.key, shape=(self.size * self.batch_size,), minval=0, maxval=num_X)
                grads_all, token = self._sample_func(lambda u, v: grad_cost_fn(self.params, u, v), X, y, batch_inxs, return_python_obj=True)

                if self.is_root_proc:
                    #self._log(f"Applying gradient updates of length {len(grads_all)}", silence)
                    for grads in grads_all:
                        # Update the gradients
                        updates, opt_state = self.optimizer.update(grads, opt_state, params=self.params)
                        self.params = optax.apply_updates(self.params, updates)

                # Send the updated weights back
                mpi4jax.barrier()  # make sure to weight for the root node to finish updating
                #self._log("Exchanging updated parameters", silence)
                self.params = self.comm.bcast(self.params, root=self.root_proc)
                #for key in self.params.keys():
                    #self.params[key], token = mpi4jax.bcast(self.params[key], root=self.root_proc, comm=self.comm,
                                                            #token=token)

            else:
                batch_inxs = jax.random.randint(key=self.key, shape=(self.batch_size,), minval=0, maxval=num_X)
                grads = grad_cost_fn(self.params, X[batch_inxs], y[batch_inxs])

                # Update the gradients
                updates, opt_state = self.optimizer.update(grads, opt_state, params=self.params)
                self.params = optax.apply_updates(self.params, updates)

            # With the updated weights, we now can calculate the current loss and accuracy of the model.
            # Again do this distributed cause we want speeeeed
            mpi4jax.barrier()

            #self._log("Calculating loss and accuracy history", silence)
            all_inxs = jnp.arange(X.shape[0] - (X.shape[0] % self.size))
            costs, token = self._sample_func(lambda u, v: cost_fn(self.params, u, v), X, y, all_inxs)
            cost = jnp.mean(costs)  # the mean of means is equivalent

            mpi4jax.barrier()

            accs, token = self._sample_func(
                lambda u, v: accuracy(v, jnp.sign(self.forward(self.params, u))),
                X, y, all_inxs)
            acc = jnp.mean(accs)

            if self.is_root_proc or (not self.use_mpi):
                loss_history.append(cost.item())
                acc_history.append(acc.item())

                if do_validation:
                    acc_test = accuracy(y_test, jnp.sign(self.forward(self.params, X_test)))
                    test_acc_history.append(acc_test.item())
                    test_loss_history.append(cost_fn(self.params, X_test, y_test).item())
                    self._log(f"epoch {epoch+1} | Cost: {cost:.4f} | Train accuracy: {acc:.4f} | Test accuracy: {acc_test:.4f}", silence)
                else:
                    self._log(f"epoch {epoch+1} | Cost: {cost:.4f} | Train accuracy: {acc:.4f}", silence)

        self.history = {
            "loss": loss_history,
            "acc": acc_history,
            "test_loss": test_loss_history if do_validation else None,
            "test_acc": test_acc_history if do_validation else None,
        }
        self._is_fitted = True
        return self

    def _sample_func(self, func, X, y, inxs, return_python_obj=False):
        # Calculate the func indexed as specific locations and return. Potentially done using a distributed setup
        # If so only self.root_proc is used for the indices
        # func(X, y) (probably a lamba function tbh)
        # if return_python_obj gather using mpi4py otherwise gather with mpi4jax
        if self.use_mpi:
            assert inxs.shape[0] % self.size == 0  # Check that the length is valid and can be split
            worker_len = inxs.shape[0] // self.size

            if self.is_root_proc:
                batch_inxs = jnp.reshape(inxs, (self.size, worker_len))
                #print(f"[{self.rank}] Reshaped indices into size of {batch_inxs.shape}")
            else:
                batch_inxs = jnp.zeros(shape=(worker_len,), dtype=jnp.int64)  # placeholder
            batch_inxs, token = mpi4jax.scatter(batch_inxs, root=self.root_proc, comm=self.comm)
            mpi4jax.barrier()

            #print(f"[{self.rank}] Sampling with indexes of {batch_inxs[:5]}... and shape {batch_inxs.shape}")
            # On each processor calculate the gradients
            z = func(X[batch_inxs], y[batch_inxs])

            if return_python_obj:
                return self.comm.gather(z, root=self.root_proc), token  # use mpi4py as just python objects
            else:
                return mpi4jax.gather(z, root=self.root_proc, comm=self.comm, token=token)
        else:
            # Just do it normally
            return func(X[inxs], y[inxs]), None

    def _forward(self, X):
        # Compute a forward pass using MPI if necessary
        if self.use_mpi:
            all_inxs = jnp.arange(X.shape[0] - (X.shape[0] % self.size))
            probs, token = self._sample_func(lambda u, v: self.forward(self.params, u), X, X, all_inxs)
            leftover = X.shape[0] % self.size
            if self.is_root_proc:
                if (leftover != 0):
                    probs_left = self.forward(self.params, X[-leftover:])
                    return jnp.hstack([probs.ravel(), probs_left])
                else:
                    return probs.ravel()
            else:
                return jnp.ones(shape=(X.shape[0],))  # all other processes can be disregarded
        else:
            return self.forward(self.params, X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X
        :param X: Input data
        :return: Predicted class probabilities
        """
        check_is_fitted(self)
        X = self.transform(X)
        probs = self._forward(X)
        return (probs + 1) / 2

    def predict(self, X):
        """
        Predict classes for X
        :param X: Input data
        :return: Predicted classes
        """
        check_is_fitted(self)
        X = self.transform(X)
        probs = self._forward(X)
        return jnp.sign(probs)

    def plot_history(self):
        """
        Plot the training loss and accuracy history.
        :return:
        """
        check_is_fitted(self)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].plot(self.history['loss'], label="Train")
        axs[1].plot(self.history['acc'])

        if self.history["test_loss"] is not None:
            axs[0].plot(self.history['test_loss'], label="Test")
            axs[1].plot(self.history['test_acc'])

        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy %')

        return fig, axs

    def plot_decision_regions(self, X, y, size=(0.0, 1.0, 0.0, 1.0), nx=40, ny=40, figsize=(7, 5), colors=['b', 'r'], color_map=plt.cm.RdBu):
        """
        Plot the decision regions for each class.
        :param X: X data to overlay
        :param y: Class labels for X
        :param size: (minx, maxx, miny, maxy)
        :param nx: number of points in x
        :param ny: number of points in y
        :return:
        """
        check_is_fitted(self)

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
        predictions_grid = self.predict(X_grid)
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
            # plot_x = (X[:, 0][Y_test == label],)
            # plot_y = (X[:, 1][Y_test == label],)
            # plt.scatter(plot_x, plot_y, c=color, marker="^", ec="k", label=f"class {label} validation")

        return fig, ax
