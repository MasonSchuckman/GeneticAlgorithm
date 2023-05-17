import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import time

def run_simulation(config, numthreads = 4):
    with open('simulations/config.json', 'w') as f:
        json.dump(config, f)
        
    start = time.time()
    subprocess.run(['./runner.exe', 'config.json', str(numthreads)])
    end = time.time()
    
    return end - start

def strong_scaling_study(base_config):
    
    num_threads = [1,2,3,4,5,6,7,8,9,10]
    times = []
    for threads in num_threads:
        times.append(run_simulation(base_config, threads))
    # num_layers = [2, 4, 6, 8, 10, 12, 14, 16]
    

    # for layers in num_layers:
    #     base_config['neural_net']['num_layers'] = layers
    #     base_config['neural_net']['layer_shapes'] = [6]*layers + [1]
    #     times.append(run_simulation(base_config))

    plt.plot(num_threads, times)
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution time (s)')
    plt.title('Strong scaling study')
    plt.grid(True)
    plt.show()

def weak_scaling_study(base_config):
    

    total_bots = [8, 9, 10, 11, 12, 13, 14]
    times = []

    for bots in total_bots:
        base_config['total_bots'] = bots
        times.append(run_simulation(base_config))
    total_bots = [2**num for num in total_bots]
    plt.plot(total_bots, times)
    plt.xlabel('Number of total bots')
    plt.ylabel('Execution time (s)')
    plt.title('Weak scaling study')
    plt.grid(True)
    plt.show()



base_config = {
        "simulation": "PongSimulation2",
        "neural_net": {
            "num_layers": 3,
            "layer_shapes": [6,32,1],
            "layer_types" : [1,0]
        },
        "bots_per_sim": 2,
        "max_iters": 100,
        "total_bots": 11,
        "num_starting_params": 9,
        "direct_contest": 1,
        "bots_per_team": 0,
        "generations" : 100,
        "base_mutation_rate" : 1,
        "min_mutation_rate" : 0.000001,
        "mutation_decay_rate" : 0.993,
        "shift_effectiveness" : 0.01,
        "load_data" : 0
    }

# Run the studies
#strong_scaling_study(base_config.copy())
weak_scaling_study(base_config.copy())