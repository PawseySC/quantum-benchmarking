# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:41 2024

Benchmark a Jacobian calculation of a circuit in PennyLane

Based on code from PennyLane's blog here: https://pennylane.ai/blog/2022/07/lightning-fast-simulations-with-pennylane-and-the-nvidia-cuquantum-sdk

@author: James
"""


import pennylane as qml
from pennylane import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def std_err(x):
    """
    Return the standard error of array[float] = std(x)/sqrt(N)
    """
    if len(x) == 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def run(wires=20, layers=3, num_runs=5, device="lightning.qubit"):
    """
    Measure time to calculate a strongly entangled circuit's jacobian in PennyLane
    :param wires: Number of qubits
    :param layers: Number of entangling layers
    :param num_runs: Number of runs to average over
    :param device: Backend device to use
    :return: average runtime and standard error
    """
    # Create the device and circuit
    dev = qml.device(device, wires=wires)
    @qml.qnode(dev, diff_method="adjoint")
    def circuit(parameters):
        qml.StronglyEntanglingLayers(weights=parameters, wires=range(wires))
        return qml.math.hstack([qml.expval(qml.PauliZ(i)) for i in range(wires)])

    # Set trainable parameters for calculating circuit Jacobian
    shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=wires)
    weights = np.random.random(size=shape)

    # Run the calculation
    timing = []
    for _ in range(num_runs):
        start = timer()
        jac = qml.jacobian(circuit)(weights)
        end = timer()
        timing.append(end - start)

    return np.mean(timing), std_err(timing)


if __name__ == "__main__":
    # Modify these following parameters to test different circuit types
    devices = ["lightning.qubit", "lightning.gpu"]
    wires = list(range(23, 30))

    results = {d: [] for d in devices}  # store average runtimes for each device
    for w in wires:
        for d in devices:
            t, err = run(wires=w, layers=3, num_runs=5, device=d)
            results[d].append(t)
            print(f"{w},{d},{t},{err}")

    # Plot the results
    for d in devices:
        plt.plot(wires, results[d], "--.", label=d)
    plt.legend()
    plt.xlabel("Number of wires")
    plt.ylabel("Time (s)")
    plt.title("Jacobian calculation runtimes")
    plt.show()
