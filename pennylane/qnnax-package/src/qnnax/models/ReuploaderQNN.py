# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:39 2025

Class for implementing a Data Re-Uploader QNN

@author: james
"""

import jax
import jax.numpy as jnp
import mpi4jax
import pennylane as qml
import optax

from qnnax.QNN import QNN


class ReuploaderQNN(QNN):
    """
    Class for implementing a Data Re-Uploader QNN.
    """
    def __init__(self,
                 dev_type="default.qubit",
                 random_state=37,
                 num_layers=3,
                 num_qubits=2,
                 num_features=2,  # num features of X
                 batch_size=128,
                 epochs=50,
                 learning_rate=0.1,
                 optimizer=optax.adam,
                 comm=None):
        """
        Initialise the ReuploaderQNN model.
        :param str dev_type: PennyLane device type (eg "default.qubit")
        :param int random_state: Seed for random number generators
        :param int num_layers: Number of re-uploader layers
        :param int num_qubits: Number of duplicate qubit lines (linked with CNOT gates)
        :param int num_features: Number of X features. Must remain constant
        :param int batch_size: Number of samples per gradient update and size of the problem distributed between processes
        :param int epochs: Number of training epochs
        :param float learning_rate: Optimizer learning rate
        :param optimizer: Optimizer class to use
        :param MPI.COMM_WORLD or None comm: If using MPI this should be MPI.COMM_WORLD otherwise None
        """
        # Make sure that all variables are explicitly set here
        super().__init__(dev_type, random_state, batch_size, epochs, learning_rate, optimizer, comm, root_proc=0)
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.num_features = num_features
        self.real_num_features = int(jnp.ceil(num_features / 3).item())*3

    def initialise(self):
        """
        Initialise the weights and bias with random values.
        """
        if self.use_mpi:
            if self.is_root_proc:
                weights = jax.random.uniform(shape=(self.num_layers, self.num_qubits, 3), key=self.key, dtype=jnp.float32)
                bias = jnp.array(0.0, dtype=jnp.float32)
            else:
                weights = jnp.zeros((self.num_layers, self.num_qubits, 3), dtype=jnp.float32)
                bias = jnp.array(0.0, dtype=jnp.float32)
            weights, token = mpi4jax.bcast(weights, root=self.root_proc, comm=self.comm)  # all other nodes should be overwritten
            bias, token = mpi4jax.bcast(bias, root=self.root_proc, comm=self.comm, token=token)

        else:
            weights = jax.random.uniform(shape=(self.num_layers, self.num_qubits, 3), key=self.key, dtype=jnp.float32)
            bias = jnp.array(0.0, dtype=jnp.float32)

        self.params = {"weights": weights, "bias": bias}
        self.create_circuit()

    def create_circuit(self):
        """
        Create the Pennylane variational circuit.
        :return: vmap version of the jitted circuit
        """
        dev = qml.device(self.dev_type, wires=self.num_qubits)
        all = range(self.num_qubits)

        # Create the variational circuit
        @qml.qnode(dev, interface="jax")
        def circuit(weights, x):
            for layer in range(self.num_layers):
                for q in all:
                    for qq in range(self.real_num_features // 3):
                        qml.Rot(*x[3*qq:3*(qq+1)], wires=q)  # Upload data (potentially with a few rotations)
                    qml.Rot(*weights[layer, q], wires=q)  # Apply weights
                if self.num_qubits > 1:  # entangle via CNOT gates
                    for q in all:
                        qml.CNOT(wires=[q, (q+1) % self.num_qubits])

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        def forward_fn(params, x):
            return circuit(params["weights"], x) + params["bias"]

        self.forward = jax.vmap(jax.jit(forward_fn), in_axes=(None, 0))
        return self.forward

    def transform(self, X: jax.Array) -> jax.Array:
        """
        Transform the input data to valid dimension ranges for the QNN.
        :param jax.Array X:  Input data of shape (n_samples, n_features)
        :return: Output data of shape (n_samples, real_num_features) where real_num_features is n_features rounded up to the nearest multiple of 3
        :rtype: jax.Array (n_samples, n_model_features)
        """
        missing_dims = self.real_num_features - X.shape[1]
        padding = jnp.zeros((X.shape[0], missing_dims), dtype=jnp.float32)
        return jnp.hstack([X, padding])
