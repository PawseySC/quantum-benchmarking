# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:27 2024

Benchmark Grover's algorithm in PennyLane

@author: James
"""

import pennylane as qml
import numpy as np
import timeit
import argparse


def std_err(x):
    """Return the standard error = std(x)/sqrt(N)"""
    if len(x) == 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def circuit(n, omegas, L):
    """Create Grover's algorithm for n qubits, searching for omegas states and repeating L times"""
    wires = list(range(n))

    # Apply the Hadamard gate to gain a state of equal superpositions
    for w in wires:
        qml.Hadamard(wires=w)

    for _ in range(L):
        for omega in omegas:
            qml.FlipSign(omega, wires=wires)  # apply the oracle
        qml.GroverOperator(wires)  # apply Grover's operator

    return [qml.expval(qml.PauliZ(i)) for i in range(n)]  # measure qubits


def benchmark(device="lightning.cpu", repeats=5):
    """Benchmark the circuit using the given device type"""
    dev = qml.device(device, wires=n)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--qubits", type=int, help="Number of qubits", default=5)
    parser.add_argument("-d", "--device", type=str, help="Device type to use", default="default.qubit")
    parser.add_argument("-l", "--repeats", type=int, help="Number of Grover operator iterations", default=10)
    args = parser.parse_args()
    n = args.qubits
    device = args.device
    L = args.repeats

    N = 2**n  # computational basis states
    omegas = np.array([
        np.ones(n)  # just search for the state of all ones
    ])

    print(f"Simulating {n} qubits with {N} computational basis states over {L} layers")
    benchmark(device, repeats=10)
