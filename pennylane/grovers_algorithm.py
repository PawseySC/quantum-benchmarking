# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:27 2024

Benchmark Grover's algorithm in PennyLane

@author: James
"""

import pennylane as qml
from pennylane import numpy as np
import timeit
import argparse


def std_err(x):
    """
    Return the standard error of array[float] = std(x)/sqrt(N)
    """
    if len(x) == 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def circuit(n, omegas, L):
    """
    Create Grover's algorithm for n qubits, searching for omegas states and repeating L times.
    Args:
        n (int): number of qubits.
        omegas (array[int]): array of states to search for.
        L (int): number of Grover iterations.
    """
    wires = list(range(n))

    # Apply the Hadamard gate to gain a state of equal superpositions
    for w in wires:
        qml.Hadamard(wires=w)

    for _ in range(L):
        for omega in omegas:
            qml.FlipSign(omega, wires=wires)  # apply the oracle
        qml.GroverOperator(wires)  # apply Grover's operator

    return [qml.expval(qml.PauliZ(i)) for i in range(n)]  # measure qubits


def benchmark(device="default.qubit", n=8, repeats=5, L=10, shots=100):
    """
    Benchmark the circuit using the given device type.
    Args:
        device (str, optional): the backend device to use. Defaults to "lightning.cpu".
        n (int, optional): number of qubits. Defaults to 8.
        repeats (int, optional): number of times to average the circuit runtime. Defaults to 5.
        L (int, optional): number of Grover operator iterations. Defaults to 10.
        shots (int, optional): number of simulation shots. Defaults to 100.
    """
    dev = qml.device(device, wires=n, shots=shots)

    omegas = np.array([  # states to search for
        np.ones(n)  # just search for the state of all ones
    ])

    @qml.qnode(dev)
    def _circuit():
        return circuit(n, omegas, L)

    runtimes = timeit.repeat(
        "_circuit()",
        setup="import pennylane as qml",
        globals={"_circuit": _circuit},
        number=1,
        repeat=repeats
    )

    time = np.mean(runtimes)
    err = std_err(runtimes)
    print(f"Runtime {device}: ({time:.4g} +- {err:.2g}) s")
    return time, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--qubits", type=int, help="Number of qubits", default=8)
    parser.add_argument("-d", "--device", type=str, help="Device type to use", default="default.qubit")
    parser.add_argument("-l", "--repeats", type=int, help="Number of Grover operator iterations", default=10)
    parser.add_argument("-s", "--shots", type=int, help="Number of shots", default=100)
    args = parser.parse_args()
    n = args.qubits
    device = args.device
    L = args.repeats
    shots = args.shots

    print(f"Simulating {n} qubits with {2**n} computational basis states over {L} layers ({shots} shots)")
    benchmark(device, n=n, repeats=10, L=L, shots=shots)
