#include "Simulator.cuh"
using std::vector;

extern __constant__ SimConfig config_d;

// Constructor allocates all necessary device memory prior to doing simulations
Simulator::Simulator(vector<Bot*> bots, Simulation *derived, SimConfig &config) : bots{bots}, config{config}, derived{derived}
{
    int totalBots = bots.size();

    int botNetSize = (config.totalNeurons + config.totalWeights); // how many indices a single bot uses in the networks_h array.

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers
    // Note: all GPU arrays are member variables.
    cudaStatus = cudaMalloc((void **)&layerShapes_d, config.numLayers * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }
    else
    {
        cudaMalloc((void **)&startingParams_d, config.numStartingParams * sizeof(float));
        cudaMalloc((void **)&output_d, totalBots * sizeof(float));

        cudaMalloc((void **)&weights_d, config.totalWeights * totalBots * sizeof(float));
        cudaMalloc((void **)&nextGenWeights_d, config.totalWeights * totalBots * sizeof(float));

        cudaMalloc((void **)&biases_d, totalBots * config.totalNeurons * sizeof(float));
        cudaMalloc((void **)&nextGenBiases_d, totalBots * config.totalNeurons * sizeof(float));


        // Copy the config over to GPU memory
        check(cudaMemcpyToSymbol(config_d, &config, sizeof(SimConfig)));

        // Setup the simulation class on the GPU
        cudaMalloc(&sim_d, sizeof(Simulation **));
        Kernels::createDerived<<<1, 1>>>(sim_d, derived->getID());
        check(cudaDeviceSynchronize());
    }
}

Simulator::~Simulator()
{
    cudaFree(layerShapes_d);
    cudaFree(startingParams_d);
    cudaFree(output_d);
    cudaFree(weights_d);
    cudaFree(biases_d);
    cudaFree(nextGenBiases_d);
    cudaFree(nextGenWeights_d);

    // Free the simulation class on the GPU
    Kernels::delete_function<<<1, 1>>>(sim_d);
    check(cudaDeviceSynchronize());

    cudaFree(sim_d);
}

void Simulator::simulate()
{
    batchSimulate(1);
}

void Simulator::formatBotData(int *&layerShapes_h, float *&startingParams_h,
                              float *&output_h, float *&weights_h, float *&biases_h)
{   
    for(int i = 0; i < config.numLayers; i++){
        layerShapes_h[i] = config.layerShapes[i];
    }
    // for(int i = 0; i < config.numStartingParams; i++){
    //     startingParams_h[i] = config[i];
    // }

    int totalBots = bots.size();
    int i = 0;
    for (const Bot *b : bots)
    {
        int WO = 0;
        int BO = 0;
        for (int j = 0; j < config.numLayers; j++)
        {
            for (int k = 0; k < layerShapes_h[j]; k++)
            {
                // set the biases
                if (j == 0)
                {
                    // input layer biases are 0
                    biases_h[i * config.totalNeurons + BO + k] = 0;
                }
                else
                {
                    // other layers get a bias = layer number.
                    biases_h[i * config.totalNeurons + BO + k] = j;
                }
                if (j != config.numLayers - 1)
                {
                    for (int l = 0; l < layerShapes_h[j + 1]; l++)
                    {
                        // set the weights. all layers get a weight of layerNum+1
                        weights_h[i * config.totalNeurons + WO + k * layerShapes_h[j + 1] + l] = j + 1;
                    }
                }
            }
            if (j != config.numLayers - 1)
            {
                BO += layerShapes_h[j];
                WO += layerShapes_h[j] * layerShapes_h[j + 1];
            }
        }

        i++;
    }
}

void Simulator::copyToGPU(int *&layerShapes_h, float *&startingParams_h,
                          float *&output_h, float *&weights_h, float *&biases_h)
{
    int totalBots = bots.size();
    check(cudaMemcpy(layerShapes_d, layerShapes_h, config.numLayers * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(startingParams_d, startingParams_h, config.numStartingParams * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output_d, output_h, totalBots * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(weights_d, weights_h, totalBots * config.totalWeights * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(biases_d, biases_h, totalBots * config.totalNeurons * sizeof(float), cudaMemcpyHostToDevice));
}
#include <chrono>

void Simulator::runSimulation(float * & output_h)
{
    int totalBots = bots.size();
    int tpb = 32; // threads per block
    int numBlocks = (totalBots / config.bpb);
    printf("Num blocks = %d\n", numBlocks);

    int sharedMemNeeded = (config.totalWeights + config.totalNeurons * 2) * config.bpb;
    printf("Shared mem needed per block = %d KB\n", sharedMemNeeded * sizeof(float) / (2 << 10));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch a kernel on the GPU with one block for each simulation/contest
    //Kernels::simulateShared<<<numBlocks, tpb, sharedMemNeeded * sizeof(float)>>>(numBlocks, this->sim_d, weights_d, biases_d, startingParams_d, output_d);
    Kernels::simulateShared_noStaticArrays<<<numBlocks, tpb, sharedMemNeeded * sizeof(float)>>>(numBlocks, this->sim_d, weights_d, biases_d, startingParams_d, output_d);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    check(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Simulation time taken: " << elapsed_time << " ms\n";

    //Mutate the genes
    float mutateMagnitude = 1.0f;

    start_time = std::chrono::high_resolution_clock::now();
    Kernels::mutate<<<numBlocks, tpb>>>(numBlocks, mutateMagnitude, weights_d, biases_d, output_d, nextGenWeights_d, nextGenBiases_d);
    check(cudaDeviceSynchronize());
    end_time = std::chrono::high_resolution_clock::now();
    
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Mutation time taken: " << elapsed_time << " ms\n";

    //swap which weights/biases arrays are "current"
    float* temp = nextGenBiases_d;
    nextGenBiases_d = biases_d;
    biases_d = temp;

    temp = nextGenWeights_d;
    nextGenWeights_d = weights_d;
    weights_d = temp;


    // Copy output vector from GPU buffer to host memory.
    //check(cudaMemcpy(output_h, output_d, totalBots * sizeof(float), cudaMemcpyDeviceToHost));
    
}

void Simulator::batchSimulate(int numSimulations)
{

    printf("num bots = %d, numLayers = %d, num weights = %d, numNeurons = %d\n", bots.size(), config.numLayers, config.totalWeights, config.totalNeurons);
    int totalBots = bots.size();

    // Allocate storage for bot data
    int *layerShapes_h = new int[config.numLayers];
    float *startingParams_h = new float[config.numStartingParams];
    float *output_h = new float[totalBots];
    float *weights_h = new float[config.totalWeights * totalBots];
    float *biases_h = new float[config.totalNeurons * totalBots];
    printf("Allocated host memory.\n");

    // Convert all the bot data to the format we need to transfer to GPU
    formatBotData(layerShapes_h, startingParams_h, output_h, weights_h, biases_h);
    printf("Formatted bot data.\n");

    // Copy it over to the GPU
    copyToGPU(layerShapes_h, startingParams_h, output_h, weights_h, biases_h);
    printf("Copied data to GPU.\n");

    // Invoke the kernel
    for(int i = 0; i < numSimulations; i++){
        runSimulation(output_h);
    }

    printf("Ran simulation.\n");

    //Do something with the output data....


    delete[] layerShapes_h;
    delete[] startingParams_h;
    delete[] output_h;
    delete[] weights_h;
    delete[] biases_h;
}

Bot *Simulator::getBest()
{
    return nullptr;
}
