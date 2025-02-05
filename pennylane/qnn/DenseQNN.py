# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:36 2025

Class to create and train a Dense (Vanilla) Quantum Neural Network (QNN)

Based upon notebook tutorials/variational_classifier.ipynb and qnn/tutorial/models/vanilla_qnn.py

Note that this is written to be run on a GPU. Jax is used by default and everything is jitted

@author: james
"""

import jax
import jax.numpy as jnp
import mpi4jax
import pennylane as qml
import optax

from QNN import QNN
from datasets import rescale


class DenseQNN(QNN):
    def __init__(self,
                 dev_type="default.qubit",
                 random_state=37,
                 num_qubits=2,
                 num_layers=6,
                 batch_size=8,
                 epochs=30,
                 learning_rate=0.01,
                 optimizer=optax.adam,
                 comm=None):
        # Make sure that all variables are explicitly set here
        super().__init__(dev_type, random_state, batch_size, epochs, learning_rate, optimizer, comm, root_proc=0)
        self.num_qubits = num_qubits
        self.num_features = 2 ** num_qubits
        self.num_layers = num_layers

    def initialise(self):
        """
        Initialise the weights and bias with random values
        """
        if self.use_mpi:
            if self.is_root_proc:
                weights = 0.01 * jax.random.uniform(shape=(self.num_layers, self.num_qubits, 3), key=self.key, dtype=jnp.float32)
                bias = jnp.array(0.0, dtype=jnp.float32)
            else:
                weights = jnp.zeros((self.num_layers, self.num_qubits, 3), dtype=jnp.float32)
                bias = jnp.array(0.0, dtype=jnp.float32)
            weights, token = mpi4jax.bcast(weights, root=self.root_proc, comm=self.comm)  # all other nodes should be overwritten
            bias, token = mpi4jax.bcast(bias, root=self.root_proc, comm=self.comm, token=token)

        else:
            weights = 0.01 * jax.random.uniform(shape=(self.num_layers, self.num_qubits, 3), key=self.key, dtype=jnp.float32)
            bias = jnp.array(0.0, dtype=jnp.float32)

        self.params = {"weights": weights, "bias": bias}
        self.create_circuit()

    def create_circuit(self):
        """
        Create the Pennylane variational circuit
        :return: vmap version of the jitted circuit
        """

        if self.dev_type == "lightning.gpu":
            dev = qml.device(self.dev_type, wires=self.num_qubits, mpi=self.use_mpi, batch_obs=self.use_mpi)  # multi-gpu
        else:
            dev = qml.device(self.dev_type, wires=self.num_qubits)

        # Create the variational circuit
        all = range(self.num_qubits)

        @qml.qnode(dev, interface="jax")
        def circuit(weights, x):
            qml.StatePrep(x, wires=all)  # amplitude encode the states
            for layer in range(self.num_layers):  # apply each layer
                for wire in all:
                    qml.Rot(*weights[layer, wire], wires=wire)  # with weights
                for q in range(self.num_qubits):
                    qml.CNOT(wires=[q, (q+1) % self.num_qubits])  # entangle qubits
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        def forward_fn(params, x):
            return circuit(params["weights"], x) + params["bias"]

        self.forward = jax.vmap(jax.jit(forward_fn), in_axes=(None, 0))
        return self.forward

    def transform(self, X):
        """
        Transform the input to valid 4D
        :param X:  Data of shape (n_samples, 2)
        :return:  Data of shape (n_samples, 4)
        """
        # Cap to range [0, 1]
        X = rescale(X, min=0, max=1)
        
        missing_dims = self.num_features - X.shape[1]
        padding = 0.1 * jnp.ones((X.shape[0], missing_dims))
        nX = jnp.hstack([X, padding])

        # Normalise each input
        normal = jnp.sqrt(jnp.sum(nX**2, -1))
        X = (nX.T / normal).T
        
        return X
