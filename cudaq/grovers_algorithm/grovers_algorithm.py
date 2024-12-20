# 20 December 2024
# Benchmarking grover's algorithm using Cudaq Python
# Process: Using 1 register of qubits

import sys
import timeit
import cudaq


# Define our kernel.
@cudaq.kernel
def kernel(qubit_count: int):
    # Superposition
    qvector = cudaq.qvector(qubit_count)
    h(qvector)

    for i in range(10):
        # Mark
        z.ctrl(qvector[:-1], qvector[-1])

        # Diffusion
        h(qvector)
        x(qvector)
        z.ctrl(qvector[:-1], qvector[-1])
        x(qvector)
        h(qvector)

    mz(qvector)


qubit_count = int(sys.argv[1]) if 1 < len(sys.argv) else 27
code_to_time = 'cudaq.sample(kernel, qubit_count, shots_count=100)'

if cudaq.num_available_gpus() > 0:
    # Execute on GPU backend.
    cudaq.set_target('nvidia')

    time = timeit.timeit(stmt=code_to_time, globals=globals(), number=1)
    print(qubit_count, time, sep=",")
