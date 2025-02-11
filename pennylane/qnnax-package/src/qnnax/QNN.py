# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 08:27 2025

Base class for a QNN model.

@author: james
"""

from typing import Self
from abc import abstractmethod

import mpi4jax
import jax
import jax.numpy as jnp
import pennylane as qml
import optax

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted


def accuracy(labels, predictions):
    return jnp.mean(jnp.equal(labels, predictions))


class QNN(ClassifierMixin, BaseEstimator):
    """
    Base class for a QNN model. All QNN models should inherit this class and implement the given methods.
    """
    @abstractmethod
    def __init__(self,
                 dev_type="default.qubit",
                 random_state=37,
                 batch_size=128,
                 epochs=30,
                 learning_rate=0.1,
                 optimizer=optax.adam,
                 comm=None,  # Pass MPI.COMM_WORLD if wishing to use MPI
                 root_proc=0):
        """
        Initialise the QNN model. Make sure all variables are initialised here.
        :param str dev_type: PennyLane device type (eg "default.qubit")
        :param int random_state: Seed for random number generators
        :param int batch_size: Number of samples per gradient update and size of the problem distributed between processes
        :param int epochs: Number of training epochs to evaluate
        :param float learning_rate: Optimizer learning rate
        :param optimizer: Optax optimizer class to use
        :param MPI.COMM_WORLD or None comm: If using MPI this should be MPI.COMM_WORLD otherwise None
        :param int root_proc: If using MPI this should be the number of the root MPI process
        """

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
        Initialise the weights and bias with random values.
        Must implement this method.
        """
        self.params = {}  # create weights here
        self.create_circuit()

    @abstractmethod
    def create_circuit(self):
        """
        Create the Pennylane variational circuit.
        Must implement this method.
        :return: The vmap version of the jitted circuit
        """
        # Demonstration code as an example
        self.circuit = None
        def forward_fn(params, x):
            # return circuit(params["weights"], x) + params["bias"]
            pass
        self.forward = jax.vmap(jax.jit(forward_fn), in_axes=(None, 0))
        return self.forward

    @abstractmethod
    def transform(self, X: jax.Array) -> jax.Array:
        """
        Transform the input data to valid ranges for the QNN both in terms of values and dimensions.
        Must implement this method.
        :param jax.Array X:  Input data of shape (n_samples, n_features)
        :return: Output data of shape (n_samples, n_model_features) potentially also normalised
        :rtype: jax.Array (n_samples, n_model_features)
        """
        return X

    def _log(self, message: str, silence=False):
        """
        Log the message with the MPI process number unless silenced.
        :param str message: Message to log
        :param bool silence: If true, do nothing
        """
        if not silence:
            if self.use_mpi:
                print(f"[{self.rank}] {message}")
            else:
                print(message)

    def fit(self, X: jax.Array, y: jax.Array, X_test=None, y_test=None, silence=False) -> Self:
        """
        Fit the QNN model on the given data.
        :param jax.Array X: Input X data of shape (n_samples, n_features)
        :param jax.Array y: Input y data of class labels (n_samples,) either -1 or 1
        :param jax.Array or None X_test: Input test X data (optional)
        :param jax.Array or None y_test: Input test y data (optional)
        :param bool silence: If true, print no training progress
        :return: Returns self
        :rtype: QNN
        """
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
                    for grads in grads_all:
                        # Update the gradients
                        updates, opt_state = self.optimizer.update(grads, opt_state, params=self.params)
                        self.params = optax.apply_updates(self.params, updates)

                # Send the updated weights back
                mpi4jax.barrier()  # make sure to weight for the root node to finish updating
                self.params = self.comm.bcast(self.params, root=self.root_proc)  # send python objects

            else:
                batch_inxs = jax.random.randint(key=self.key, shape=(self.batch_size,), minval=0, maxval=num_X)
                grads = grad_cost_fn(self.params, X[batch_inxs], y[batch_inxs])

                # Update the gradients
                updates, opt_state = self.optimizer.update(grads, opt_state, params=self.params)
                self.params = optax.apply_updates(self.params, updates)

            # With the updated weights, we now can calculate the current loss and accuracy of the model.
            # Again do this distributed cause we want speeeeed
            mpi4jax.barrier()

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

    def _sample_func(self, func, X: jax.Array, y: jax.Array, inxs: jax.Array, return_python_obj=False):
        """
        Internal method to sample a function at many points while distributing it between multiple MPI processes (if available).
        :param func: Function to sample with parameters `func(X[inxs], y[inxs])`
        :param jax.Array X: Input X data to sample over
        :param jax.Array y: Input y class labels to sample over
        :param jax.Array inxs: Indicies to sample at. Only indices sent to the root process 
        :param bool return_python_obj: If true gathers the result of `func` with `mpi4py` for python objects or uses `mpi4jax` for Jax arrays
        """
        if self.use_mpi:
            assert inxs.shape[0] % self.size == 0  # Check that the length is valid and can be split
            worker_len = inxs.shape[0] // self.size

            if self.is_root_proc:
                batch_inxs = jnp.reshape(inxs, (self.size, worker_len))
            else:
                batch_inxs = jnp.zeros(shape=(worker_len,), dtype=jnp.int32)  # placeholder
            batch_inxs, token = mpi4jax.scatter(batch_inxs, root=self.root_proc, comm=self.comm)
            mpi4jax.barrier()

            # On each processor calculate the gradients
            z = func(X[batch_inxs], y[batch_inxs])

            if return_python_obj:
                return self.comm.gather(z, root=self.root_proc), token  # use mpi4py as just python objects
            else:
                return mpi4jax.gather(z, root=self.root_proc, comm=self.comm, token=token)
        else:
            # Just do it normally
            return func(X[inxs], y[inxs]), None

    def _forward(self, X: jax.Array) -> jax.Array:
        """
        Compute a forward pass of the circuit.
        :param jax.Array X: Input data X
        :return: The forward pass
        :rtype: jax.Array
        """
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

    def predict_proba(self, X: jax.Array) -> jax.Array:
        """
        Predict the class probabilities for some input data.
        :param jax.Array X: Input data X
        :return: Predicted class probabilities
        :rtype: jax.Array
        """
        check_is_fitted(self)
        X = self.transform(X)
        probs = self._forward(X)
        return (probs + 1) / 2

    def predict(self, X: jax.Array) -> jax.Array:
        """
        Predict each class for some input data.
        :param jax.Array X: Input data X
        :return: Predicted classes
        :rtype: jax.Array
        """
        check_is_fitted(self)
        X = self.transform(X)
        probs = self._forward(X)
        return jnp.sign(probs)
