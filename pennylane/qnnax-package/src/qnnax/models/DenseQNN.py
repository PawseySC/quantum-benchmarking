# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:36 2025

Class to create a Dense Quantum Neural Network (QNN)

@author: james
"""

import jax
import jax.numpy as jnp
import mpi4jax
import pennylane as qml
import optax

from qnnax.QNN import QNN
from qnnax.datasets import rescale


class DenseQNN(QNN):
    """
    Class to create a Dense QNN.
    """
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
        """
        Initialise the DenseQNN model.
        :param str dev_type: PennyLane device type (eg "default.qubit")
        :param int random_state: Seed for random number generators
        :param int num_qubits: Number of qubit lines (Data is extended to fill the full Hilbert space so can support 2^(num_qubits) X features)
        :param int num_layers: Number of weighted layers
        :param int batch_size: Number of samples per gradient update and size of the problem distributed between processes
        :param int epochs: Number of training epochs
        :param float learning_rate: Optimizer learning rate
        :param optimizer: Optimizer class to use
        :param MPI.COMM_WORLD or None comm: If using MPI this should be MPI.COMM_WORLD otherwise None
        """
        # Make sure that all variables are explicitly set here
        super().__init__(dev_type, random_state, batch_size, epochs, learning_rate, optimizer, comm, root_proc=0)
        self.num_qubits = num_qubits
        self.num_features = 2 ** num_qubits
        self.num_layers = num_layers

    def initialise(self):
        """
        Initialise the weights and bias with random values.
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
        Create the Pennylane variational circuit.
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

    def transform(self, X: jax.Array) -> jax.Array:
        """
        Transform the input data to valid ranges for the QNN both in terms of values and dimensions.
        :param jax.Array X:  Input data of shape (n_samples, n_features)
        :return: Output data of shape (n_samples, real_num_features) where real_num_features is 2**num_qubits and normalised
        :rtype: jax.Array (n_samples, n_model_features)
        """
        # Cap to range [0, 1]
        X = rescale(X, min=0, max=1)
        
        missing_dims = self.num_features - X.shape[1]
        padding = 0.1 * jnp.ones((X.shape[0], missing_dims), dtype=jnp.float32)
        nX = jnp.hstack([X, padding])

        # Normalise each input
        normal = jnp.sqrt(jnp.sum(nX**2, -1))
        X = (nX.T / normal).T
        return X
