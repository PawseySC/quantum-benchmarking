# Quax (*Q-nix*)
> MPI enabled quantum machine learning for PennyLane and Jax

Over 10x speedup for Quantum Machine Learning Algorithms!

[github.com/gyndlf/qnnax](https://github.com/gyndlf/qnnax)

## Installation

Depending on the backend desired to use packages must be installed (or even built manually) carefully.

- First install Jax with GPU/CUDA support
- Then install MPI support for Jax (mpi4jax)
- Finally install PennyLane and lightning.gpu from source (otherwise it doesn't seem to build right)

```bash
# Install Jax
$ pip install -U "jax[cuda12]"

# Install mpi4jax
$ pip install cython
$ pip install mpi4py
$ pip install mpi4jax --no-build-isolation

# Build PennyLane from source
$ git clone https://github.com/PennyLaneAI/pennylane-lightning.git
$ cd pennylane-lightning/
$ git reset --hard
$ pip install -r requirements.txt -vv
$ pip install custatevec-cu12

$ export CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum')")
$ PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
$ CMAKE_ARGS="-DENABLE_MPI=ON" pip install -e . --config-settings editable_mode=compat -vv
$ PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
$ CMAKE_ARGS="-DENABLE_MPI=ON" pip install -e . --config-settings editable_mode=compat -vv

# Grab some extra packages
$ pip install notebook
$ pip install optax
$ pip install scikit-learn pandas matplotlib
```

To quickly test that the environment is built right run the following

```python
import jax
jax.devices()  # should show CUDA devices (if available)

import pennylane as qml
dev = qml.device("lightning.gpu", wires=2)  # should give no errors
dev = qml.device("lightning.cpu", wires=2)  # should give no errors
```

## Usage

For a minimal example see the following code.

```python
import qnnax

qnn = qnnax.ReuploaderQNN(num_qubits=2, num_layers=4, num_features=4, batch_size=512)

X, y = qnnax.datasets.create_circles(seed=0, n=1000, num_features=4)

qnn.fit(X, y)

print(qnn.score(X, y))

>> 0.872
```

For an MPI example see `run.py`. 

## Creating a custom QNN

1. Create a custom class and inherit `qnnax.QNN`
2. Implement methods of
    - `qnn.transform()` to take arbitrary data and reshape it for the circuit (normalisation, adding extra dimensions, etc)
    - `qnn.initialise()` to randomly create weights used in the circuit. This *must* save the weights in the dictionary `qnn.params = {}`. 
This should also call `qnn.create_circuit()` to build the PennyLane circuit using the weights previously generated.
This method *must* set the forward (using `jax.vmap`) function of `qnn.forward()` to simulate the circuit.
3. Fit and run as normal with `qnn.fit()` and `qnn.predict()`.



