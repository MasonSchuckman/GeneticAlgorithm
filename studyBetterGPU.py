import subprocess
import time
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# Function to run the MPI program
def run_simulation(config, num_trials):
    execution_times = np.zeros(num_trials)
    with open('simulations/config.json', 'w') as f:
        json.dump(config, f)
    for i in range(num_trials):
        start_time = time.time()
        subprocess.run(['./runner.exe', 'config.json'])
        end_time = time.time()
        execution_times[i] = end_time - start_time
    return np.mean(execution_times)


# Check if saved data file exists
def load_saved_data():
    if os.path.isfile('simulations/test_results.npy'):
        return np.load('simulations/test_results.npy')
    else:
        return None


# Save test results
def save_test_results(results):
    os.makedirs('simulations', exist_ok=True)
    np.save('simulations/test_results.npy', results)


base_config = {
    "simulation": "TargetSimulation",
    "neural_net": {
        "num_layers": 3,
        "layer_shapes": [6, 32, 4],
        "layer_types": [0, 0]
    },
    "bots_per_sim": 1,
    "max_iters": 300,
    "total_bots": 1024,
    "num_starting_params": 5,
    "direct_contest": -1,
    "bots_per_team": 0,
    "generations": 100,
    "base_mutation_rate": 0.01,
    "min_mutation_rate": 0.000001,
    "mutation_decay_rate": 0.998,
    "shift_effectiveness": 0.5,
    "load_data": 0
}

# Test parameters
array_sizes = []
for i in range(6):
    array_sizes.append(2 ** (i + 6))

num_hidden_layers = [1, 2, 3, 4, 5, 6]
num_trials = 3

use_saved = False

if use_saved:
    # Load saved data or run the tests
    saved_data = load_saved_data()
else:
    saved_data = None

if saved_data is not None:
    execution_times = saved_data
else:
    # Run the MPI program and store the execution times
    execution_times = np.zeros((len(array_sizes), len(num_hidden_layers)))

    for i, size in enumerate(array_sizes):
        cur_config = base_config.copy()
        cur_config['total_bots'] = size
        for j, hidden_layers in enumerate(num_hidden_layers):
            cur_config['neural_net']['num_layers'] = hidden_layers
            cur_config['neural_net']['layer_shapes'] = [6] + [16] * hidden_layers + [4]
            execution_times[i, j] = run_simulation(cur_config, num_trials)

    save_test_results(execution_times)
# Calculate speedup and efficiency on a per-array-size basis
speedup = np.zeros_like(execution_times)
efficiency = np.zeros_like(execution_times)
for i in range(len(array_sizes)):
    speedup[i] = execution_times[i, 0] / execution_times[i]
    efficiency[i] = speedup[i] / (i + 6)

print(execution_times)
# First Plot: Network Depth vs. Runtime
plt.figure(figsize=(8, 6))
for i, size in enumerate(array_sizes):
    plt.plot(num_hidden_layers, execution_times[i, :], marker='o', label=f"Array Size: {size}")

plt.xlabel("Network Depth")
plt.ylabel("Runtime (s)")
plt.title("Network Depth vs. Runtime")
plt.legend(title="Number of Hidden Layers")
plt.grid(True)
plt.show()

# Second Plot: Array Size vs. Runtime
plt.figure(figsize=(8, 6))
for i, hidden_layers in enumerate(num_hidden_layers):
    plt.plot(array_sizes, execution_times[:, i], marker='o', label=f"Network Depth: {hidden_layers}")

plt.xlabel("Array Size")
plt.ylabel("Runtime (s)")
plt.title("Array Size vs. Runtime")
plt.legend(title="Population Size")
plt.grid(True)
plt.show()
