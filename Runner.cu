#include "SimulationList.cuh"
#include "Kernels.cuh"


#include "Simulation.cuh"
#include "Simulator.cuh"
#include "TargetSimulation.cuh"

#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>

#include <fstream>
#include "json.hpp"
using json = nlohmann::json;


using std::vector;

void getNetInfo(int &numConnections, int &numNeurons, std::vector<int> layerShapes)
{
    // Calculate how many connections and neurons there are based on layerShapes_h so we can create the networks_h array.
    for (int i = 0; i < layerShapes.size(); i++)
    {
        if (i != layerShapes.size() - 1)
            numConnections += layerShapes[i] * layerShapes[i + 1]; // effectively numWeights
        numNeurons += layerShapes[i];                                // effectively numBiases
    }
}

FullSimConfig setupSimulation(const std::string& filename) {
    std::ifstream file(filename);
    json configFile;

    // Parse the JSON file
    try {
        file >> configFile;
    } catch (const json::parse_error& e) {
        std::cerr << "Failed to parse config file " << filename << ": " << e.what() << std::endl;
        exit(1);
    }

    // Read the simulation type from the configFile
    std::string simType = configFile["simulation"].get<std::string>();
    Simulation * sim;
    if (simType == "TargetSimulation") {
        sim = new TargetSimulation;
    } else if (simType == "typeB") {
        //sim = TargetSimulation::TypeB;
    } else {
        std::cerr << "Unknown simulation type: " << simType << std::endl;
        exit(1);
    }

    // Read the neural net configuration from the configFile
    int numLayers = configFile["neural_net"]["num_layers"].get<int>();    
    std::vector<int> layerShapes = configFile["neural_net"]["layer_shapes"].get<std::vector<int>>();
    int numConnections = 0;
    int numNeurons = 0;
    getNetInfo(numConnections, numNeurons, layerShapes);

    // Read the rest of the simulation configuration from the config
    int botsPerSim = configFile["bots_per_sim"].get<int>();
    int maxIters = configFile["max_iters"].get<int>();

    //Note: the totalBots we put in the json is log_2 of what we simulate.
    int totalBots = configFile["total_bots"].get<int>();
    totalBots = (int) std::pow(2, totalBots);

    int numStartingParams = configFile["num_starting_params"].get<int>();
    int directContest = configFile["direct_contest"].get<int>();
    int botsPerTeam = configFile["bots_per_team"].get<int>();

    int generations = configFile["generations"].get<int>();

    float baseMutationRate = configFile["base_mutation_rate"].get<float>();
    float minMutationRate = configFile["min_mutation_rate"].get<float>();
    float mutationDecayRate = configFile["mutation_decay_rate"].get<float>();

    SimConfig config(numLayers, numNeurons, numConnections, botsPerSim, maxIters, numStartingParams, directContest, botsPerTeam);
    for(int i = 0; i < layerShapes.size(); i++)
        config.layerShapes[i] = layerShapes[i];

    // Create and return the SimConfig object
    return FullSimConfig(sim, config, totalBots, generations, baseMutationRate, minMutationRate, mutationDecayRate);
}

// Define constant GPU memory for the config of our simulation.
// Note: This CAN be set at runtime
__constant__ SimConfig config_d;

void launchKernel(Simulation *derived, SimConfig &config)
{

    cudaSetDevice(0);

    // Create the sim on the GPU.
    Simulation **sim_d;
    cudaMalloc(&sim_d, sizeof(Simulation **));

    // Copy the config over to GPU memory
    check(cudaMemcpyToSymbol(config_d, &config, sizeof(SimConfig)));

    Kernels::createDerived<<<1, 1>>>(sim_d, derived->getID());
    check(cudaDeviceSynchronize());

    int n = 6;
    Kernels::game_kernel<<<2, 3>>>(n, sim_d);
    check(cudaDeviceSynchronize());

    Kernels::delete_function<<<1, 1>>>(sim_d);
    check(cudaDeviceSynchronize());

    cudaFree(sim_d);

    printf("done\n");
    // Code to launch the CUDA kernel with the configured parameters and function pointer
}



void test_simulation_1()
{
    // Define which simulation we're running
    BasicSimulation sim;

    // Define the neural net configuration for our bots
    int numLayers = 3;
    int numConnections = 0, numNeurons = 0;
    vector<int> layerShapes;
    layerShapes.push_back(8);
    layerShapes.push_back(32);
    layerShapes.push_back(8);
      
    getNetInfo(numConnections, numNeurons, layerShapes);


    // Define the rest of the simulation configuration
    int botsPerSim = 1;

    if(botsPerSim > MAX_BOTS_PER_SIM){
        printf("Increase MAX_BOTS_PER_SIM and run again.\n");
        
    }

    int maxIters = 5;
    int totalBots = 1 << 15;
    int numStartingParams = 1;
    int directContest = 0;
    int botsPerTeam = 0;

    SimConfig config(numLayers, numNeurons, numConnections, botsPerSim, maxIters, numStartingParams, directContest, botsPerTeam);
    for (int i = 0; i < numLayers; i++)
    {
        config.layerShapes[i] = layerShapes[i];
    }
    
    vector<Bot*> bots;
    for(int i = 0; i < totalBots; i++){
        bots.push_back(new Bot(layerShapes));
    }

    printf("Created bots.\n");


    Simulator engine(bots, &sim, config);

    engine.batchSimulate(1);

    
    for(int i = 0; i < totalBots; i++){
        delete bots[i];
    }

    

}

void test_simulation_2()
{
    FullSimConfig fullConfig = setupSimulation("TargetSimConfig.json");

    vector<Bot*> bots;
    for(int i = 0; i < fullConfig.totalBots; i++){
        bots.push_back(new Bot(fullConfig.config.layerShapes, fullConfig.config.numLayers));
    }

    Simulator engine(bots, fullConfig.sim, fullConfig.config);
    engine.min_mutate_rate = fullConfig.minMutationRate;
    engine.mutateMagnitude = fullConfig.baseMutationRate;
    engine.mutateDecayRate = fullConfig.mutationDecayRate;
    engine.batchSimulate(fullConfig.generations);

    
    for(int i = 0; i < fullConfig.totalBots; i++){
        delete bots[i];
    }
}


int main()
{   
    cudaSetDevice(0);

    auto start_time = std::chrono::high_resolution_clock::now();
    test_simulation_2();    
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Total time taken: " << elapsed_time << " ms\n";

    // cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

    return 0;
}