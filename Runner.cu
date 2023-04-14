#include "Simulation.cuh"
#include "BasicSimulation.cuh"
#include "Kernels.cuh"



#include <iostream>


#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU error check: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Define constant GPU memory for the config of our simulation. 
// Note: This CAN be set at runtime
__constant__ SimConfig config_d;



void launchKernel(Simulation* derived, SimConfig & config) {
   
    cudaSetDevice(0);

    //Create the sim on the GPU.
    Simulation** sim_d;
    cudaMalloc(&sim_d, sizeof(Simulation**));

    //Copy the config over to GPU memory
    check(cudaMemcpyToSymbol(config_d, &config, sizeof(SimConfig)));

    Kernels::createDerived<<<1,1>>>(sim_d, derived->getID());
    check(cudaDeviceSynchronize());

    int n = 6;
	Kernels::game_kernel<<<2, 3>>>(n, sim_d);
	check(cudaDeviceSynchronize());

    Kernels::delete_function<<<1,1>>>(sim_d);
    check(cudaDeviceSynchronize());

    cudaFree(sim_d);
    

	printf("done\n");
    // Code to launch the CUDA kernel with the configured parameters and function pointer
}

int main() {
    BasicSimulation sim;
    int botsPerSim = 1;
    int numLayers = 3;
    int totalBots = 1024 * 32;
    int numStartingParams = 1;
    int *layerShapes_h = new int[numLayers];
    float *startingParams_h = new float[numStartingParams];
    float *output_h = new float[totalBots];
    float *weights_h;
    float *biases_h;

    int numConnections = 0;
    int numNeurons = 0;

    // Determine the network configuration
    layerShapes_h[0] = 8;
    layerShapes_h[1] = 32;
    layerShapes_h[2] = 8;

    // Calculate how many connections and neurons there are based on layerShapes_h so we can create the networks_h array.
    for (int i = 0; i < numLayers; i++)
    {
        if (i != numLayers - 1)
            numConnections += layerShapes_h[i] * layerShapes_h[i + 1]; // effectively numWeights
        numNeurons += layerShapes_h[i];                                // effectively numBiases
    }

    int maxIters = 10;
    SimConfig config{numLayers, numNeurons, numConnections, botsPerSim, maxIters};
    for(int i = 0; i < numLayers; i++){
        config.layerShapes[i] = layerShapes_h[i];
    }

    launchKernel(&sim, config);

    delete [] layerShapes_h;
    delete [] startingParams_h;
    delete [] output_h;

    return 0;
}