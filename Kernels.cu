#include "SimulationList.cuh"
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>

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
    __device__ void forward_propagation(const float *inputs, const float *weights, const float *biases, float *output, int input_size, int output_size, int layer)
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
#pragma unroll
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

        __syncthreads();

        //  Apply activation function
        switch (config_d.layerTypes[layer])
        {
        // linear
        case 0:
            break;

        // ReLU
        case 1:
        {
            for (int i = tid; i < output_size; i += stride)
                output[i] = output[i] > 0 ? output[i] : 0; // max(output[i],0)
        }
        break;

        // Sigmoid
        case 2:
        {
            for (int i = tid; i < output_size; i += stride)
                output[i] = 1.0f / (1.0f + expf(-output[i]));
        }
        break;

        // Default is linear
        default:
            break;
        }

        __syncthreads();
    }

    // This is a basic example for what the simulation kernel will look like.
    __global__ void game_kernel(int n, Simulation **sim)
    {

        // Placeholder for our activations/gamestate arrays.
        __shared__ float s1[64];
        __shared__ float s2[64];
        float *botActivs[MAX_BOTS_PER_SIM];
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

    using namespace cub;   
    __device__ float blockReduceSum(float val)
    {
        __syncthreads();
        // Specialize BlockReduce for a 1D block of 128 threads of type int
        typedef cub::BlockReduce<float, 32> BlockReduceT;
        // Allocate shared memory for BlockReduce
        __shared__ typename BlockReduceT::TempStorage temp_storage;
        
        // Compute the block-wide sum for thread0
        float aggregate = BlockReduceT(temp_storage).Sum(val);

        return aggregate;
    }

    template <typename T>
    __device__ T block_reduce(T *input, int n)
    {
        int tid = threadIdx.x;

        // Allocate shared memory for temp sums
        extern __shared__ T sdata[];

        __syncthreads();

        // Perform reduction within each thread's assigned range
        float threadSum = 0;
        for (int i = tid; i < n; i += blockDim.x)
        {
            threadSum += fabsf(input[i]);
        }

        sdata[tid] = threadSum;

        __syncthreads();

        // Perform reduction across threads in the block
        // do reduction in shared mem
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads(); // make sure all adds at one stage are done!
        }

        return sdata[0];
    }

    __device__ void zeroArray(float *arr, int length)
    {
        for (int i = threadIdx.x; i < length; i++)
        {
            arr[i] = 0;
        }
        __syncthreads();
    }

    // Each block will go through each layer of its respective bot(s), and threads will edit individual weights/biases.
    // The nextGenWeights/biases arrays are the exact same shape and size of the allWeights/biases arrays, but with the genetic information of the next generation.
    __global__ void mutate(const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome, int *childSpecies,
                           float *nextGenWeights, float *nextGenBiases, float *distances, float *deltas, int *ancestors, float progThreshold, const int gen, const int shift)
    {
        int gid = threadIdx.x + blockIdx.x * blockDim.x; // global id
        int tid = threadIdx.x;                           // thread id (within a block)

        int block = blockIdx.x;
        int stride = blockDim.x;

        // prevent OOB errors
        if (block < n / 2)
        {
            curandState_t state;
            curand_init(blockIdx.x + shift, threadIdx.x, 0, &state);

            float rand = 0;

            // calcuate the offset for this block's bot(s)
            int offsetBot1 = block * 2;
            int offsetBot2 = (block * 2 + shift * 2 + 1) % n;

            int botOffsets[2] = {offsetBot1, offsetBot2};

            float botScore1 = simulationOutcome[offsetBot1];
            float botScore2 = simulationOutcome[offsetBot2];
            if (botScore1 == 0 && botScore2 == 0 && threadIdx.x == 0)
                printf("Error. Both zero. block = %d, offset1 = %d, offset2 = %d\n", blockIdx.x, offsetBot1, offsetBot2);

            int winnerBotOffset;
            if (botScore1 > botScore2)
            {
                winnerBotOffset = offsetBot1;
            }
            else
            {
                winnerBotOffset = offsetBot2;
            }

            // Only need 1 thread updating these. Don't want conflicts.
            if (tid == 0)
            {
                // keeping track of the parent specimen from which the children came from
                childSpecies[offsetBot1] = winnerBotOffset;
                childSpecies[offsetBot2] = winnerBotOffset;

                // Update the ancestor for the bots (only the losing bot gets updated tbh)
                ancestors[offsetBot1] = ancestors[winnerBotOffset];
                ancestors[offsetBot2] = ancestors[winnerBotOffset];
            }

            __syncthreads();

            float distance = 0;       // Distance from the parent
            float deltaMagnitude = 0; // deltaMagnitude is the L1 norm of a bot's genome and the progenitor it decended from.

            for (int bot = 0; bot < 2; bot++)
            {
                // Write this bot's updated weights
                for (int i = tid; i < config_d.totalWeights; i += stride)
                {
                    if (bot == 1) // Only add noise to bot 1 (bot 0 stays the same as the winner)
                        rand = curand_uniform(&state) * randomMagnitude * 2 - randomMagnitude;
                    distance += fabsf(rand);

                    (deltas)[i + botOffsets[bot] * config_d.paddedNetworkSize] += rand;
                    (nextGenWeights)[i + botOffsets[bot] * config_d.totalWeights] = (allWeights)[i + winnerBotOffset * config_d.totalWeights] + rand;
                }

                // Write this bot's updated biases
                // We can skip the first layer since the input layer shouldn't have biases.
                for (int i = tid + config_d.layerShapes[0]; i < config_d.totalNeurons; i += stride)
                {
                    if (bot == 1) // Only add noise to bot 1 (bot 0 stays the same as the winner)
                        rand = curand_uniform(&state) * randomMagnitude * 2 - randomMagnitude;
                    distance += fabsf(rand);

                    (deltas)[i + config_d.totalWeights + botOffsets[bot] * config_d.paddedNetworkSize] += rand;
                    (nextGenBiases)[i + botOffsets[bot] * config_d.totalNeurons] = (allBiases)[i + winnerBotOffset * config_d.totalNeurons] + rand;
                }

                float totalDistance = blockReduceSum(distance);
                
                //if (tid == 0)
                //{
                    (distances)[botOffsets[bot]] = totalDistance;
                    deltaMagnitude = block_reduce<float>(&(deltas[botOffsets[bot] * config_d.paddedNetworkSize]), config_d.paddedNetworkSize);

                    // if (blockIdx.x == 0)
                    //     printf("delta mag = %f\n", deltaMagnitude);
                    // Check if child is a new species
                    if (deltaMagnitude >= progThreshold)
                    {
                        (ancestors)[botOffsets[bot]] = botOffsets[bot] + gen * n;

                        // Reset the deltas for this bot since it is now the prog
                        zeroArray(&(deltas[botOffsets[bot] * config_d.paddedNetworkSize]), config_d.paddedNetworkSize);
                    }
                //}
                __syncthreads();
            }
        }

        __syncthreads();

        return;
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
            case 2:
                (*sim) = new TargetSimulation();
                break;
            case 3:
                (*sim) = new MultibotSimulation();
                break;
            case 4:
                (*sim) = new PongSimulation();
                break;
            case 5:
                (*sim) = new AirHockeySimulation();
                break;
            case 6:
                (*sim) = new PongSimulation2();
                break;
            case 7:
                (*sim) = new MultiBallPong();
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

    // this kernel divides the work into blocks rather than each thread works alone.
    // Reason for doing this is to hopefully make better use of cache and reduce memory stalls.
    // (Hopefully this will lead to higher FLOPS).

    __device__ int counter = 0;
    __global__ void simulateShared2(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output)
    {

        int gid = threadIdx.x + blockIdx.x * blockDim.x; // global id
        int tid = threadIdx.x;                           // thread id (within a block)

        int block = blockIdx.x;
        int stride = blockDim.x;

        // prevent OOB errors
        if (block < n)
        {

            // hard coding this makes things *much* simpler. We can change it if needed.
            __shared__ float gamestate[64];
            (*sim)->setupSimulation(startingParams, gamestate);

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

            __syncthreads();
            // Copy this block's weights and biases to the shared arrays.
            for (int i = tid; i < config_d.totalWeights * config_d.bpb; i += stride)
            {
                weights[i] = (allWeights)[block * config_d.totalWeights * config_d.bpb + i];
            }
            for (int i = tid; i < config_d.totalNeurons * config_d.bpb; i += stride)
            {
                biases[i] = (allBiases)[block * config_d.totalNeurons * config_d.bpb + i];
            }

            __syncthreads();

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
                // Set the activations for this bot this iteration
                (*sim)->setActivations(gamestate, activs, iter);
                __syncthreads();

                // It's important to remember that activs and ws and bs are essentially 2d arrays. That's why indexing them is tricky and weird.
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

                        // forward_propagation(float* input, float* weights, float* biases, float* output, int input_size, int output_size)
                        forward_propagation(activs[bot] + AO, ws[bot] + WO, bs[bot] + numBiases + BO, activs[bot] + AO + numBiases, numBiases, config_d.layerShapes[layer + 1], layer);

                        // This register-less version had almost identical performance
                        // forward_propagation(activs(bot) + numBiases * layer, ws(bot) + numWeights * layer, bs(bot) + numBiases * (layer + 1), activs(bot) + numBiases * (layer + 1), numBiases, config_d.layerShapes[layer + 1]);

                        AO += numBiases;
                        WO += numWeights;
                        BO += numBiases;
                    }
                }

                // update simulation/game state based on bot actions
                (*sim)->eval(actions, gamestate);

                if ((*sim)->checkFinished(gamestate) == 1)
                {
                    finished = true;
                }

                __syncthreads();
                iter++;
                if (iter >= maxIters || finished)
                {
                    finished = true;
                    (*sim)->setOutput(output, gamestate, startingParams);
                    __syncthreads();
                    // if(threadIdx.x == 0 && output[blockIdx.x * 2] == 0 || output[blockIdx.x * 2 + 1] == 0){
                    //     printf("block %d is zero at iter %f\n", blockIdx.x, startingParams[8]);
                    // }
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

    // TODO: compare performance of this vs simulateShared2
    __global__ void simulateShared_noStaticArrays(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output)
    {

        int gid = threadIdx.x + blockIdx.x * blockDim.x; // global id
        int tid = threadIdx.x;                           // thread id (within a block)

        int block = blockIdx.x;
        int stride = blockDim.x;

        // prevent OOB errors
        if (block < n)
        {

            // hard coding this makes things *much* simpler. We can change it if needed.
            __shared__ float gamestate[64];
            // hard coding for TargetSimulation
            if (tid == 0)
            {
                gamestate[0] = 0;
                gamestate[1] = 0;
                gamestate[2] = 0;
                gamestate[3] = 0;
                gamestate[4] = 0;
                output[block] = 0;
            }
            __syncthreads();
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

            if (gid == 0)
            {
                printf("Weights inside kernel (%d): \n", config_d.totalWeights);
                for (int i = 0; i < 4; i++)
                {
                    printf("%f, ", (allWeights)[block * config_d.totalWeights + i]);
                }
                printf("\n");
            }
            __syncthreads();
            // Copy this block's weights and biases to the shared arrays.
            for (int i = tid; i < config_d.totalWeights * config_d.bpb; i += stride)
            {
                weights[i] = (allWeights)[block * config_d.totalWeights + i];
            }
            for (int i = tid; i < config_d.totalNeurons * config_d.bpb; i += stride)
            {
                biases[i] = (allBiases)[block * config_d.totalNeurons + i];
            }

#define ws(i) weights + config_d.totalWeights *i
#define bs(i) biases + config_d.totalNeurons *i
#define activs(i) activations + config_d.totalNeurons *i
            // #define actions(i) config_d.totalNeurons - config_d.layerShapes[config_d.numLayers - 1] + activs(i);

            float *actions[MAX_BOTS_PER_SIM];

            // // Populate the arrays created above
            for (int i = 0; i < config_d.bpb; i++)
            {
                //     ws[i] = weights + config_d.totalWeights * i;
                //     bs[i] = biases + config_d.totalNeurons * i;
                //     activs[i] = activations + config_d.totalNeurons * i;

                //     // TODO: check if this correctly offsets actions[i] to point to the last layer of bot_i's activations network.
                actions[i] = activs(i) + config_d.totalNeurons - config_d.layerShapes[config_d.numLayers - 1];
            }

            __syncthreads();

            int maxIters = config_d.maxIters;
            bool finished = false;

            int iter = 0; // current timestep of simulation we're on

            // run the simulation loop.
            while (!finished)
            {
                // Do something with the starting parameters here. Possibly call a sim function.
                // if (iter == 0)
                // {

                //     // Determine inputs for the bot(s)
                //     for (int i = 0; i < config_d.bpb; i++)
                //     {
                //         for (int j = tid; j < config_d.layerShapes[0]; j += stride)
                //         {
                //             // This line is a placeholder for now.
                //             (activs(i))[j] = 0.0f;

                //         }

                //     }

                //     if(tid == 0){
                //         (activs(0))[0] = 0.0f;
                //     }
                // }
                // hard coding for TargetSimulation:
                const int numInputs = 4;
                if (tid < numInputs)
                {
                    (activs(0))[tid] = gamestate[tid];
                }

                // It's important to remember that activs and nns are essentially 2d arrays. That's why indexing them is tricky and weird.
                // Poll the NN for actions.

                __syncthreads();
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

                        // forward_propagation(float* input, float* weights, float* biases, float* output, int input_size, int output_size)
                        forward_propagation(activs(bot) + AO, ws(bot) + WO, bs(bot) + numBiases + BO, activs(bot) + AO + numBiases, numBiases, config_d.layerShapes[layer + 1], layer);

                        // This register-less version had almost identical performance
                        // forward_propagation(activs(bot) + numBiases * layer, ws(bot) + numWeights * layer, bs(bot) + numBiases * (layer + 1), activs(bot) + numBiases * (layer + 1), numBiases, config_d.layerShapes[layer + 1]);

                        AO += numBiases;
                        WO += numWeights;
                        BO += numBiases;
                    }
                }
                // printf("about to eval\n");

                // update simulation/game state based on bot actions
                (*sim)->eval(&activations, gamestate);

                if ((*sim)->checkFinished(gamestate) == 1)
                {
                    finished = true;
                }

                __syncthreads();
                iter++;
                if (iter >= maxIters)
                {
                    finished = true;
                    if (tid == 0)
                        output[block] = -gamestate[4];
                    if (block == 0 && tid == 0)
                    {
                        printf("Block %d total dist = %f\n", blockIdx.x, gamestate[4]);
                    }
                    __syncthreads();
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

};