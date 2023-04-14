#include "Simulation.cuh"
#include "BasicSimulation.cuh"
#include <curand_kernel.h>

extern __constant__ SimConfig config_d;

namespace Kernels
{
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
    __global__ void simulateShared(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output)
    {

        int gid = threadIdx.x + blockIdx.x * blockDim.x; // global id
        int tid = threadIdx.x;                           // thread id (within a block)

        int block = blockIdx.x;
        int stride = blockDim.x;

        // prevent OOB errors
        if (block < n)
        {   

            //hard coding this makes things *much* simpler. We can change it if needed.
            __shared__ float gamestate[64]; 


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

            int maxIters = config_d.maxIters;
            bool finished = false;

            int iter = 0; // current timestep of simulation we're on

            // run the simulation loop.
            while (!finished)
            {
                // Do something with the starting parameters here. Possibly call a sim function.
                if (iter == 0)
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
                (*sim)->eval(actions, gamestate);
                
                if((*sim)->checkFinished(gamestate) == 1){
                    finished = true;
                }

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


    //This is a basic example for what the simulation kernel will look like.
    __global__ void game_kernel(int n, Simulation **sim)
    {

        // Placeholder for our activations/gamestate arrays.
        __shared__ float s1[64];
        __shared__ float s2[64];
        float * botActivs[MAX_BOTS_PER_SIM];
        botActivs[0] = s1;

        // global id
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid == 0)
        {
            printf("max iters = %d\n", config_d.maxIters);
        }
        // block-local id
        int tid = threadIdx.x;

        // Avoid OOB errors
        if (gid < n)
        {
            for (int i = 0; i < blockIdx.x + 2; i++)
            {
                (*sim)->eval(botActivs, s2);
                __syncthreads();
            }
        }
        return;
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

    __global__ void createDerived(Simulation **sim, int id)
    {
        // From https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions
        // It is necessary to create object representing a function
        // directly in global memory of the GPU device for virtual
        // functions to work correctly, i.e. virtual function table
        // HAS to be on GPU as well.

        // NOTE: THIS SWITCH STATEMENT MUST BE UPDATED EACH TIME WE ADD A NEW SIMULATION!
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            switch (id)
            {
            case 1:
                (*sim) = new BasicSimulation();
                break;

            default:
                printf("Invalid derived class ID. Did you update the kernel switch statement?\n");
                break;
            }
        }

        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid == 0)
        {
            printf("total neurons = %d\n", config_d.totalNeurons);
        }

        return;
    };

    __global__ void delete_function(Simulation **sim)
    {
        delete *sim;
    };
};