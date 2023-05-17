import subprocess
import time
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# Function to run the MPI program
def run_simulation(config, num_processes, num_trials):
    execution_times = np.zeros(num_trials)
    with open('simulations/config.json', 'w') as f:
        json.dump(config, f)
    for i in range(num_trials):
        start_time = time.time()
        subprocess.run(['./runner.exe', 'config.json', str(num_processes)])
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
    "total_bots": 10,
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
num_processes = [1, 2, 3, 4, 5, 6, 7, 8]
num_trials = 3

use_saved = True

if use_saved:
    # Load saved data or run the tests
    saved_data = load_saved_data()
else:
    saved_data = None

if saved_data is not None:
    execution_times = saved_data
else:
    # Run the MPI program and store the execution times
    execution_times = np.zeros((len(array_sizes), len(num_processes)))

    for i, size in enumerate(array_sizes):
        cur_config = base_config.copy()
        cur_config['total_bots'] = size
        for j, processes in enumerate(num_processes):
            execution_times[i, j] = run_simulation(cur_config, processes, num_trials)

    save_test_results(execution_times)

# Calculate speedup and efficiency on a per-array-size basis
speedup = np.zeros_like(execution_times)
efficiency = np.zeros_like(execution_times)
for i in range(len(array_sizes)):
    speedup[i] = execution_times[i, 0] / execution_times[i]
    efficiency[i] = speedup[i] / np.array(num_processes)

combined_plot = False

if combined_plot:
    # Plot the results
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    # Main Title
    fig.suptitle("Performance Analysis of the Genetic Algorithm on CPU", fontsize=16)

    # Plot 1: Execution time
    for i, size in enumerate(array_sizes):
        axs[0].plot(num_processes, execution_times[i], label=f"Population size: {size}")

    axs[0].set_xlabel("Number of Processes")
    axs[0].set_ylabel("Execution Time (s)")
    axs[0].set_title("Scalability")
    axs[0].set_xticks(num_processes)
    axs[0].legend()
    axs[0].grid()

    # Plot 2: Speedup
    for i, size in enumerate(array_sizes):
        axs[1].plot(num_processes, speedup[i], marker='o', label=f"Population size: {size}")

    axs[1].set_xlabel("Number of Processes")
    axs[1].set_ylabel("Speedup")
    axs[1].set_title("Speedup")
    axs[1].set_xticks(num_processes)
    axs[1].legend()
    axs[1].grid()

    # Plot 3: Efficiency
    for i, size in enumerate(array_sizes):
        axs[2].plot(num_processes, efficiency[i], marker='o', label=f"Population size: {size}")

    axs[2].set_xlabel("Number of Processes")
    axs[2].set_ylabel("Efficiency")
    axs[2].set_title("Efficiency")
    axs[2].set_xticks(num_processes)
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()

else:
    # Plot 1: Execution time
    plt.figure(figsize=(8, 6))
    plt.plot(num_processes, execution_times.T)
    plt.xlabel("Number of Processes")
    plt.ylabel("Execution Time (s)")
    plt.title("Scalability")
    plt.xticks(num_processes)
    plt.legend(array_sizes, title="Population size")
    plt.grid(True)
    plt.show()

    # Plot 2: Speedup
    plt.figure(figsize=(8, 6))
    plt.plot(num_processes, speedup.T, marker='o')
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup")
    plt.title("Speedup")
    plt.xticks(num_processes)
    plt.legend(array_sizes, title="Population size")
    plt.grid(True)
    plt.show()

    # Plot 3: Efficiency
    plt.figure(figsize=(8, 6))
    plt.plot(num_processes, efficiency.T, marker='o')
    plt.xlabel("Number of Processes")
    plt.ylabel("Efficiency")
    plt.title("Efficiency")
    plt.xticks(num_processes)
    plt.legend(array_sizes, title="Population size")
    plt.grid(True)
    plt.show()

