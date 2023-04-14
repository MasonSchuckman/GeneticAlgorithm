#include "Simulation.cuh"
#include "BasicSimulation.cuh"
#include "Kernels.cuh"


#include "Simulation.cuh"
#include "Simulator.cuh"

#include <iostream>
#include <vector>
#include <chrono>

using std::vector;

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

void getNetInfo(int &numConnections, int &numNeurons, int numLayers, int *& layerShapes)
{
    // Calculate how many connections and neurons there are based on layerShapes_h so we can create the networks_h array.
    for (int i = 0; i < numLayers; i++)
    {
        if (i != numLayers - 1)
            numConnections += layerShapes[i] * layerShapes[i + 1]; // effectively numWeights
        numNeurons += layerShapes[i];                                // effectively numBiases
    }
}

void test_simulation_1()
{
    // Define which simulation we're running
    BasicSimulation sim;

    // Define the neural net configuration for our bots
    int numLayers = 3;
    int numConnections = 0, numNeurons = 0;
    int *layerShapes = new int[numLayers];
    layerShapes[0] = 8;
    layerShapes[1] = 32;
    layerShapes[2] = 8;    
    getNetInfo(numConnections, numNeurons, numLayers, layerShapes);


    // Define the rest of the simulation configuration
    int botsPerSim = 1;
    int maxIters = 500;
    int totalBots = 1 << 15;
    int numStartingParams = 1;
    
    SimConfig config(numLayers, numNeurons, numConnections, botsPerSim, maxIters, numStartingParams);
    for (int i = 0; i < numLayers; i++)
    {
        config.layerShapes[i] = layerShapes[i];
    }
    
    vector<Bot*> bots;
    for(int i = 0; i < totalBots; i++){
        bots.push_back(new Bot(layerShapes, numLayers));
    }

    printf("Created bots.\n");


    Simulator engine(bots, &sim, config);

    engine.batchSimulate(2);

    for(int i = 0; i < totalBots; i++){
        delete bots[i];
    }

    delete [] layerShapes;

}

int main()
{   
    cudaSetDevice(0);

    auto start_time = std::chrono::high_resolution_clock::now();
    test_simulation_1();    
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