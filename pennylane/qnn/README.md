# Build Instructions

- First install Jax with GPU/CUDA support
- Then install MPI support for Jax
- Then install pennylane and lightning.gpu from source (otherwise it doesn't seem to build right??)

```bash
$ module load py-3.12.4-cuquantum/24.03.0
$ module load cmake
$ module load ninja
$ export CXX=g++
$ export CC=gcc
$ export FC=gfortran

$ python -m venv qnn_mpi --system-site-packages
$ source qnn_mpi/bin/activate

$ pip install -U pip
$ pip install -U "jax[cuda12]"
$ pip install cython
$ pip install mpi4py
$ pip install mpi4jax --no-build-isolation

$ cd pennylane-lightning/
$ git reset --hard
$ pip install -r requirements.txt -vv
$ pip install custatevec-cu12
$ export CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum')")
$ PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
$ CMAKE_ARGS="-DENABLE_MPI=ON" pip install -e . --config-settings editable_mode=compat -vv
$ PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
$ CMAKE_ARGS="-DENABLE_MPI=ON" pip install -e . --config-settings editable_mode=compat -vv

$ pip install notebook
$ pip install optax
$ pip install scikit-learn pandas matplotlib
```

Test the environment with

```python
import jax
jax.devices()  # should show CUDA devices

import pennylane as qml
dev = qml.device("lightning.gpu", wires=2)
```

# Run the model

```bash
srun -n 4 python run.py
```
