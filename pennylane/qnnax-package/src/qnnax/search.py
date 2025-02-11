# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:10 2025

Searcher class to optimise the circuit hyperparameters (number of layers, qubits, etc)

@author: james
"""

from itertools import product
from sklearn.model_selection import train_test_split
import optax
from mpi4py import MPI

import qnnax

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

is_root = rank == 0


class Searcher:
    def __init__(self, seed=37):
        """
        Initialise the hyperparameter searcher
        :param int seed: Random seed
        """
        self.seed = seed
        self.datasets = None
        self.models = None
        self.results = {}

    def set_datasets(self, datasetss: dict):
        """
        Set the datasets to evaluate for.
        :param dict(str, tuple) datasets: Dictionary of datasets
        """
        for name, (X, y) in datasets.items():
            assert X.shape[0] == y.shape[0]
            assert type(name) is str
        self.datasets = datasets

    def set_models(self, models: dict):
        """
        Set the models to evaluate with.
        :param list models: Models to evaluate
        """
        for model in models:
            assert len(model) == 2
            assert issubclass(model[0], qnnax.QNN)
            assert type(model[1]) is dict
        self.models = models

    def search_hyperparameters(self, verbose=True) -> dict:
        """
        Search the given hyperparameters to determine the model with the greatest accuracy.
        :param bool verbose: If true, then print the progress
        :return: Results from the hyperparameter searching
        :rtype: dict
        """
        for name, (X, y) in self.datasets.items():
            if verbose: print(f"#----- {name} -----#")
            for (Qnn, params) in self.models:
                product_values = product(*[v if isinstance(v, (list, tuple)) else [v] for v in params.values()])
                all_args = [dict(zip(params.keys(), values)) for values in product_values]
                if verbose: print(f"> Searching for hyperparameters for {Qnn.__name__} through {len(all_args)} combinations.")

                best_acc = 0.0
                best_params = None
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
                for arg in all_args:
                    qnn = Qnn(**arg)
                    acc = qnn.fit(X_train, y_train, silence=True).score(X_test, y_test)
                    if verbose: print(f"{qnn.get_params()} -> {acc * 100:.2f} %")
                    if acc > best_acc:
                        best_acc = acc
                        best_params = arg
                if verbose: print(f"Best hyperparameters for {Qnn.__name__} [{best_acc * 100:.2f}%]: {best_params}\n")
                self.results[(name, Qnn)] = best_acc, best_params

        # Print the results
        if is_root: print("#----------#")
        for (dataset_name, Qnn), (acc, params) in self.results.items():
            qnn_name = Qnn.__name__
            remaining_length = 32 - (len(dataset_name) + len(qnn_name))
            if is_root: print(f"{dataset_name} [{qnn_name}] {" " * remaining_length} {acc * 100:.2f} % \t| {params}")
        if is_root: print("#----------#")
        return self.results


if __name__ == "__main__":
    seed = 37
    n = 10000
    n_qubits = 2

    # Use MPI if available
    if size == 1:
        comm = None
    else:
        comm = COMM

    datasets = {
        "classification": qnnax.create_classification(seed, n),
        "circles": qnnax.create_circles(seed, n, num_features=2**n_qubits)
    }

    models = [
       [qnnax.ReuploaderQNN, {
           "dev_type": "default.qubit",
           "num_layers": [3,4,5,6],
           "batch_size": [32,64,128],
           "learning_rate": [0.1, 0.01],
           "comm": comm,
       }],
        [qnnax.DenseQNN, {
            "dev_type": "default.qubit",
            "num_qubits": n_qubits,
            "num_layers": [3, 4, 5, 6],
            "batch_size": [32, 64],
            "learning_rate": 0.1,
            "optimizer": optax.adam,
            "comm": comm,
        }]
    ]

    # Test it out!
    searcher = Searcher(seed=seed)
    searcher.set_datasets(datasets)
    searcher.set_models(models)

    results = searcher.search_hyperparameters()

