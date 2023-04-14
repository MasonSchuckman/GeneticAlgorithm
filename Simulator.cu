
#include "cuda_runtime.h"
#include "math.h"
#include <math.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <curand_kernel.h>

#include "Agent.h"
#include "Simulation.cuh"
#include "BasicSimulation.cuh"
#include "Kernels.cuh"

#include <iostream>
#include <vector>
using std::vector;

#define check(ans)                            \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU error check: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

extern __constant__ SimConfig config_d;

class Simulator
{
public:
    // Constructor allocates all necessary device memory prior to doing simulations
    Simulator(vector<Agent> agents, SimConfig config) : agents{agents}, config{config}
    {
        int totalBots = agents.size();

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
            cudaMalloc((void **)&biases_d, totalBots * config.totalNeurons * sizeof(float));
        }
    }

    ~Simulator()
    {
        cudaFree(layerShapes_d);
        cudaFree(startingParams_d);
        cudaFree(output_d);
        cudaFree(weights_d);
        cudaFree(biases_d);
    }

    void simulate(int itersPerSimulation)
    {
        batchSimulate(1, itersPerSimulation);
    }

    void batchSimulate(int numSimulations, int totalBots)
    {

        // int numStartingParams = 1;
        // int *layerShapes_h = new int[config.numLayers];
        // float *startingParams_h = new float[numStartingParams];
        // float *output_h = new float[totalBots];
        // float *weights_h;
        // float *biases_h;

        // int numConnections = 0;
        // int numNeurons = 0;

        // // Determine the network configuration
        // layerShapes_h[0] = 8;
        // layerShapes_h[1] = 32;
        // layerShapes_h[2] = 8;

        // // Calculate how many connections and neurons there are based on layerShapes_h so we can create the networks_h array.
        // for (int i = 0; i < config.numLayers; i++)
        // {
        //     if (i != config.numLayers - 1)
        //         numConnections += layerShapes_h[i] * layerShapes_h[i + 1]; // effectively numWeights
        //     numNeurons += layerShapes_h[i];                                // effectively numBiases
        // }
        // int botNetSize = (numConnections + numNeurons); // how many indices a single bot uses in the networks_h array.
        // weights_h = new float[numConnections * totalBots];
        // biases_h = new float[numNeurons * totalBots];

        // printf("Total network size = %d KB\n", numConnections * sizeof(float) / (2 << 10));

        // // initialize networks_h with random stuff for testing.

        // for (int i = 0; i < totalBots; i++)
        // {
        //     // printf("bot %d\n", i);
        //     int WO = 0;
        //     int BO = 0;
        //     for (int j = 0; j < numLayers; j++)
        //     {
        //         // printf("\tlayer %d\n", j);
        //         for (int k = 0; k < layerShapes_h[j]; k++)
        //         {
        //             // printf("\t\tNode %d: ", k);
        //             // set the biases
        //             if (j == 0)
        //             {
        //                 // input layer biases are 0
        //                 biases_h[i * numNeurons + BO + k] = 0;
        //             }
        //             else
        //             {
        //                 // other layers get a bias = layer number.
        //                 biases_h[i * numNeurons + BO + k] = j;
        //             }
        //             // printf("bias = %f, weights: ", networks_h[i * botNetSize + LO + k]);
        //             if (j != numLayers - 1)
        //             {
        //                 for (int l = 0; l < layerShapes_h[j + 1]; l++)
        //                 {
        //                     // set the weights. all layers get a weight of layerNum+1
        //                     weights_h[i * numConnections + WO + k * layerShapes_h[j + 1] + l] = j + 1;
        //                     // printf("%f, ", networks_h[i * botNetSize + LO + layerShapes_h[j] + k * layerShapes_h[j + 1] + l]);
        //                 }
        //             }

        //             // printf("\n");
        //         }
        //         if (j != numLayers - 1)
        //         {
        //             BO += layerShapes_h[j];
        //             WO += layerShapes_h[j] * layerShapes_h[j + 1];
        //         }
        //         // printf("\n");
        //     }
        //     // printf("\n");
        // }
    }

    Agent getBest()
    {
    }

private:
    /*
    Pointers to device memory should be private and persist as we're doing simulation.
    This way we don't need to constantly re allocate device memory.
    */
    int *layerShapes_d;
    float *startingParams_d;
    float *output_d;
    float *weights_d;
    float *biases_d;

    vector<Agent> agents;
    SimConfig config;

    /**
     * Perform forward propagation of a dense neural network
     *
     * @param input     input data to the network, a float array of size input_size
     * @param weights   weight matrix of the network, a float array of size input_size * output_size
     * @param biases    bias vector of the network, a float array of size output_size
     * @param output    output of the network, a float array of size output_size
     * @param input_size    size of the input data
     * @param output_size   size of the output data
     */
    __device__ void forward_propagation(const float *inputs, const float *weights, const float *biases, float *output, int input_size, int output_size)
    {
        int stride = blockDim.x;
        int tid = threadIdx.x;
#ifdef DEBUG
        if (threadIdx.x == 0)
        {
            printf("Biases : ");
            for (int i = 0; i < output_size; i++)
            {
                printf("%f, ", biases[i]);
            }
            printf("\n");
        }
#endif
        // Initialize output to biases
        for (int i = threadIdx.x; i < output_size; i += stride)
        {
            output[i] = biases[i];
        }

        // Compute dot product of input and weights
#pragma unroll 4
        for (int i = 0; i < input_size; i++)
        {
            for (int j = tid; j < output_size; j += stride)
            {
                output[j] += inputs[i] * weights[i * output_size + j];
            }
        }

#ifdef DEBUG
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {

            printf("Activs : ");
            for (int i = 0; i < output_size; i++)
                printf("%f, ", output[i]);
            printf("\n\n");
        }
#endif // DEBUG

        // TODO: Look into using different activation functions for different layers. (output should probably be sigmoid, others maybe ReLU)
        //  Apply activation function (sigmoid in this case)
        __syncthreads();
        for (int i = tid; i < output_size; i += stride)
            output[i] = 1.0f / (1.0f + expf(-output[i]));
    }

    // this kernel divides the work into blocks rather than each thread works alone.
    // Reason for doing this is to hopefully make better use of cache and reduce memory stalls.
    // (Hopefully this will lead to higher FLOPS).
    __global__ void simulateShared(const int n, const float *allWeights, const float *allBiases, const float *startingParams, float *output)
    {

        int gid = threadIdx.x + blockIdx.x * blockDim.x; // global id
        int tid = threadIdx.x;                           // thread id (within a block)

        int block = blockIdx.x;
        int stride = blockDim.x;

        // prevent OOB errors
        if (block < n)
        {

            // shared mem layout is w1,w2...,w_bpt,b1,b2,...,b_bpt,a_1,a_2,...,a_bpt
            // declare our block of shared memory
            extern __shared__ float s[];

            // split our shared memory block into the weights, biases, and activations
            float *weights = s;
            float *biases = weights + config_d.totalWeights * config_d.bpb;
            float *activations = biases + config_d.totalNeurons * config_d.bpb;

#ifdef DEBUG
            printf("Weights = %p\n", weights);
            printf("biases  = %p\n", biases);
            printf("activs  = %p\n", activations);
#endif

            // Copy this block's weights and biases to the shared arrays.
            for (int i = 0; i < config_d.totalWeights * config_d.bpb; i += stride)
            {
                weights[i] = (allWeights)[block * config_d.totalWeights + i];
            }
            for (int i = 0; i < config_d.totalNeurons * config_d.bpb; i += stride)
            {
                biases[i] = (allBiases)[block * config_d.totalNeurons + i];
            }

            // Seperate the bot(s) data
            const float *ws[MAX_BOTS_PER_SIM]; // abbreviation for "bot weights"
            const float *bs[MAX_BOTS_PER_SIM]; // abbreviation for "bot biases"
            float *activs[MAX_BOTS_PER_SIM];   // abbreviation for "bot activations"

            // The pointers in this array point to the last layer of each bots' neural net.
            // This makes it easier to pass the bots' actions' for each iteration.
            float *actions[MAX_BOTS_PER_SIM];

            // Populate the arrays created above
            for (int i = 0; i < config_d.bpb; i++)
            {
                ws[i] = weights + config_d.totalWeights * i;
                bs[i] = biases + config_d.totalNeurons * i;
                activs[i] = activations + config_d.totalNeurons * i;

                // TODO: check if this correctly offsets actions[i] to point to the last layer of bot_i's activations network.
                actions[i] = activs[i] + config_d.totalNeurons - config_d.layerShapes[config_d.numLayers - 1];
            }
            __syncthreads();

            int maxIters = 1000;
            bool finished = false;

            int iter = 0; // current timestep of simulation we're on

            // run the simulation loop.
            while (!finished)
            {
                // Determine inputs for the bot(s)
                for (int i = 0; i < config_d.bpb; i++)
                {
                    for (int j = tid; j < config_d.layerShapes[0]; j += stride)
                    {
                        // This line is a placeholder for now.
                        activs[i][j] = 0.5f;
                    }
                }

                // It's important to remember that activs and nns are essentially 2d arrays. That's why indexing them is tricky and weird.
                // Poll the NN for actions.
                for (int bot = 0; bot < config_d.bpb; bot++)
                {
                    // All of these offsets are to account for the multiple layers in the network.
                    int WO = 0; // weights offset
                    int BO = 0; // biases offset
                    int AO = 0; // activs offset
                    int numBiases;
                    int numWeights;
                    for (int layer = 0; layer < config_d.numLayers - 1; layer++)
                    {
                        numBiases = config_d.layerShapes[layer];
                        numWeights = numBiases * config_d.layerShapes[layer + 1];
#ifdef DEBUG
                        if (tid == 0)
                        {
                            printf("Weights of layer %d:\n", layer);
                            for (int k = 0; k < numBiases; k++)
                            {
                                for (int l = 0; l < layerShapes[layer + 1]; l++)
                                {
                                    printf("%f, ", (nns[i] + LO + numBiases)[k * layerShapes[layer + 1] + l]);
                                }
                                printf("\n");
                            }
                            int cc = 0;
                            for (int k = 0; k < numWeights; k++)
                            {
                                printf("%f, ", (nns[i] + LO + numBiases)[k]);
                                cc++;
                                if (cc % layerShapes[layer + 1] == 0)
                                    printf("\n");
                            }
                            printf("\n");
                        }

#endif
                        // forward_propagation(float* input, float* weights, float* biases, float* output, int input_size, int output_size)
                        forward_propagation(activs[bot] + AO, ws[bot] + WO, bs[bot] + numBiases + BO, activs[bot] + AO + numBiases, numBiases, config_d.layerShapes[layer + 1]);

                        AO += numBiases;
                        WO += numWeights;
                        BO += numBiases;
                    }
                }

                // update simulation/game state based on bot actions

                // do simulation/game logic

                // if(checkWinCondition(<something>)
                //	finished = true;

                iter++;
                if (iter >= maxIters)
                {
                    finished = true;
                }
            }
#ifdef DEBUG
            if (tid == 0 && blockIdx.x == 0)
            {
                printf("Activations:\n");
                int AO = 0; // "activs offset"
                for (int layer = 0; layer < numLayers; layer++)
                {
                    printf("Layer %d, size = %d, AO = %d\n", layer, layerShapes[layer], AO);
                    for (int i = 0; i < layerShapes[layer]; i++)
                    {
                        printf("%f, ", activs[0][AO + i]);
                    }
                    AO += layerShapes[layer];
                    printf("\n");
                }
                printf("\n");
            }
#endif
        }
        return;
    }

    // TODO: Implement function to look at the simulation's outcome array for this block and determine who won.
    //  This will be dependent on the simulation we're currently using.
    __device__ int determineWinner(float *outcome)
    {
        return 0;
    }

    // Each block will go through each layer of its respective bot(s), and threads will edit individual weights/biases.
    // The nextGenWeights/biases arrays are the exact same shape and size of the allWeights/biases arrays, but with the genetic information of the next generation.
    __global__ void mutate(const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome,
                           float *nextGenWeights, float *nextGenBiases)
    {
        int gid = threadIdx.x + blockIdx.x * blockDim.x; // global id
        int tid = threadIdx.x;                           // thread id (within a block)

        int block = blockIdx.x;
        int stride = blockDim.x;

        // prevent OOB errors
        if (block < n)
        {
            curandState_t state;
            curand_init(blockIdx.x, threadIdx.x, 0, &state);

            float rand = curand_uniform(&state) * randomMagnitude * 2 - randomMagnitude;

            // Copy this block's weights and biases to the shared arrays.
            for (int i = 0; i < config_d.totalWeights * config_d.bpb; i += stride)
            {
                // weights[i] = (allWeights)[block * totalWeights + i];
            }
            for (int i = 0; i < config_d.totalNeurons * config_d.bpb; i += stride)
            {
                // biases[i] = (allBiases)[block * totalNeurons + i];
            }
        }
    }
};