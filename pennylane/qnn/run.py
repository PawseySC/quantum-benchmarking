# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:06 2025

Just run the given model and return how long it takes

@author: james
"""


import jax.numpy as jnp

from mpi4py import MPI
import mpi4jax
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import optax
from math import log2, ceil

from DenseQNN import DenseQNN
from ReuploaderQNN import ReuploaderQNN
from datasets import create_classification, create_blobs, create_moons, create_circles


COMM = MPI.COMM_WORLD
size = COMM.Get_size()
rank = COMM.Get_rank()


if __name__ == "__main__":
    seed = 37
    n = 1000  # num datapoints
    n_features = 3
    n_qubits = ceil(log2(n_features)) + 1

    X, y = create_circles(seed, n, num_features=n_features)
    # X, y = create_classification(seed, n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    if size > 1:  # only use MPI if we actually have it available
        comm = COMM
    else:
        comm = None

    mpi4jax.barrier()

    qnn = DenseQNN(
        dev_type="default.qubit",
        num_layers=6,
        batch_size=64,
        learning_rate=0.1,
        optimizer=optax.adam,
        num_qubits=n_qubits,
        epochs=30,
        comm=comm,
    )

    # Train the model
    tic = timer()
    qnn.fit(X_train, y_train, silence=False)
    toc = timer()
    train_time = toc - tic

    mpi4jax.barrier()

    # Ok, now time how long the forward pass takes
    times = []
    for _ in range(10):  # average
        tic = timer()
        qnn.predict(X_train)
        toc = timer()

        times.append(toc - tic)
        mpi4jax.barrier()

    times = jnp.array(times[1:])  # ignore first time as jiting the circuit

    mpi4jax.barrier()
    val = qnn.score(X_test, y_test)
    mpi4jax.barrier()
    train = qnn.score(X_train, y_train)

    if rank == 0:
        print(size, jnp.mean(times), jnp.median(times), jnp.std(times), val, train)


