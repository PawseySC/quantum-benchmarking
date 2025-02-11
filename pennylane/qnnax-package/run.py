# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:06 2025

Demonstration code for running the QNN either single threaded or using MPI.

@author: james
"""


import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")

from mpi4py import MPI
import mpi4jax
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import optax
import time

import qnnax

COMM = MPI.COMM_WORLD
size = COMM.Get_size()
rank = COMM.Get_rank()


if __name__ == "__main__":
    seed = 37
    n = 10**3

    n_qubits = 5 # 11 use more qubits to make the problem harder
    n_features = 2**(n_qubits-1)

    X, y = qnnax.create_circles(seed, n, num_features=n_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    if size > 1:  # only use MPI if we actually have it available
        comm = COMM
    else:
        comm = None

    mpi4jax.barrier()

    # Choose the model to use
    #qnn = qnnax.DenseQNN(
    qnn = qnnax.ReuploaderQNN(
        dev_type="default.qubit",
        num_layers=6,
        batch_size=10,
        learning_rate=0.1,
        optimizer=optax.adam,
        num_qubits=n_qubits,
        num_features=n_features,  # only for reuploader
        epochs=10,
        comm=comm,
    )

    # Train the model
    qnn.fit(X_train, y_train, silence=False)
    mpi4jax.barrier()

    # Ok, now time how long the forward pass takes
    times = []
    for _ in range(5):  # average
        tic = timer()
        qnn.predict(X_train)
        toc = timer()

        times.append(toc - tic)

        time.sleep(1)
        mpi4jax.barrier()

    times = jnp.array(times[1:])  # ignore first time as jiting the circuit

    mpi4jax.barrier()
    val = qnn.score(X_test, y_test)
    mpi4jax.barrier()
    train = qnn.score(X_train, y_train)

    if rank == 0:
        print(size, jnp.mean(times), jnp.median(times), jnp.std(times), val, train)

