import subprocess
import time
import json
import os

import numpy as np
import matplotlib.pyplot as plt

num_threads = 7

# Function to run the MPI program
def run_simulation(config, num_trials):
    execution_times = np.zeros(num_trials)
    with open('simulations/config.json', 'w') as f:
        json.dump(config, f)
    for i in range(num_trials):
        start_time = time.time()
        subprocess.run(['./runner.exe', 'config.json', str(num_threads)])
        end_time = time.time()
        execution_times[i] = end_time - start_time
    return np.mean(execution_times)


# Check if saved data file exists
def load_saved_data():
    if os.path.isfile('simulations/test_results_netSize.npy'):
        return np.load('simulations/test_results_netSize.npy')
    else:
        return None


# Save test results
def save_test_results(results):
    os.makedirs('simulations', exist_ok=True)
    np.save('simulations/test_results_netSize.npy', results)


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
    "direct_contest": 0,
    "bots_per_team": 0,
    "generations": 1000,
    "base_mutation_rate": 0.01,
    "min_mutation_rate": 0.000001,
    "mutation_decay_rate": 0.998,
    "shift_effectiveness": 0.5,
    "load_data": 0
}

# Test parameters
num_hidden_layers = [1, 2, 3, 4, 5, 6, 7, 8]
hidden_layer_sizes = [4, 8, 12, 16, 20, 24, 28, 32]  # increments of 4, starting at 4 and ending at 32
#hidden_layer_sizes = [4, 8, 12, 16]  # increments of 4, starting at 4 and ending at 32
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
    execution_times = np.zeros((len(num_hidden_layers), len(hidden_layer_sizes)))

    cur_config = base_config.copy()
    for i, hidden_layers in enumerate(num_hidden_layers):
        cur_config['neural_net']['num_layers'] = hidden_layers
        for j, layer_size in enumerate(hidden_layer_sizes):
            cur_config['neural_net']['layer_shapes'] = [6] + [layer_size] * hidden_layers + [4]
            execution_times[i, j] = run_simulation(cur_config, num_trials)

    save_test_results(execution_times)


# First Plot: Network Depth vs. Runtime for different hidden layer sizes
fig = plt.figure(figsize=(12, 6))  # Increase the figure size to accommodate both plots

# Plot 1 - Network Depth vs. Runtime
plt.subplot(1, 2, 1)  # Create a subplot for the first plot
fig.suptitle("Performance Analysis of Network Size on CPU", fontsize=16)

for i, layer_size in enumerate(hidden_layer_sizes):
    plt.plot(num_hidden_layers, execution_times[:, i], marker='o', label=f"Hidden Layer Size: {layer_size}")

plt.xlabel("Network Depth")
plt.ylabel("Runtime (s)")
plt.title("Network Depth vs. Runtime")
plt.legend(title="Hidden Layer Size")
plt.grid(True)

# Second Plot: Hidden Layer Size vs. Runtime for different network depths
plt.subplot(1, 2, 2)  # Create a subplot for the second plot
for i, hidden_layers in enumerate(num_hidden_layers):
    plt.plot(hidden_layer_sizes, execution_times[i, :], marker='o', label=f"Network Depth: {hidden_layers}")

plt.xlabel("Hidden Layer Size")
plt.ylabel("Runtime (s)")
plt.title("Hidden Layer Size vs. Runtime")
plt.legend(title="Network Depth")
plt.grid(True)

plt.tight_layout()  # Adjust the spacing between the plots
plt.show()

desired_dpi = 300

# Save the plot with the specified resolution
fig.savefig("plot.png", dpi=desired_dpi)