#include "Simulation.cuh"
#include "SimulationList.cuh"
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

        // TODO: Look into using different activation functions for different layers. (output should probably be sigmoid, others maybe ReLU)
        __syncthreads();

        //  Apply activation function (sigmoid in this case)

        // use ReLU for non output layers
        if (layer != config_d.numLayers - 2)
        {
            for (int i = tid; i < output_size; i += stride)
                output[i] = output[i] > 0 ? output[i] : 0; // max(output[i],0)

            // use sigmoud for the output layer
        }
        else
        {

            for (int i = tid; i < output_size; i += stride)
                output[i] = 1.0f / (1.0f + expf(-output[i]));
        }

        __syncthreads();
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

            // hard coding this makes things *much* simpler. We can change it if needed.
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
                __syncthreads();
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

                        // forward_propagation(float* input, float* weights, float* biases, float* output, int input_size, int output_size)
                        forward_propagation(activs[bot] + AO, ws[bot] + WO, bs[bot] + numBiases + BO, activs[bot] + AO + numBiases, numBiases, config_d.layerShapes[layer + 1], layer);

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

                // if(checkWinCondition(<something>)
                //	finished = true;

                iter++;
                if (iter >= maxIters)
                {
                    finished = true;
                }
            }

        }
        return;
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

    // Each block will go through each layer of its respective bot(s), and threads will edit individual weights/biases.
    // The nextGenWeights/biases arrays are the exact same shape and size of the allWeights/biases arrays, but with the genetic information of the next generation.
    __global__ void mutate(const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome,
                           float *nextGenWeights, float *nextGenBiases, const int shift)
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

            float rand;

            // calcuate the offset for this block's bot(s)
            int offsetBot1 = block * 2;
            int offsetBot2 = (block * 2 + shift * 2 + 1) % n;

            float botScore1 = simulationOutcome[offsetBot1];
            float botScore2 = simulationOutcome[offsetBot2];

            int winnerBotOffset;
            if (botScore1 > botScore2)
            {
                winnerBotOffset = offsetBot1;
            }
            else
            {
                winnerBotOffset = offsetBot2;
            }

            
            __syncthreads();
            // Write next gen bot one's data
            for (int i = tid; i < config_d.totalWeights; i += stride)
            {
                rand = curand_uniform(&state) * randomMagnitude * 2 - randomMagnitude;
                (nextGenWeights)[i + offsetBot1 * config_d.totalWeights] = (allWeights)[i + winnerBotOffset * config_d.totalWeights];// + rand;
            }
            // We can skip the first layer since the input layer shouldn't have biases.
            for (int i = tid + config_d.layerShapes[0]; i < config_d.totalNeurons; i += stride)
            {
                rand = curand_uniform(&state) * randomMagnitude * 2 - randomMagnitude;
                (nextGenBiases)[i + offsetBot1 * config_d.totalNeurons] = (allBiases)[i + winnerBotOffset * config_d.totalNeurons];// + rand;
            }

            // Write next gen bot two's data
            for (int i = tid; i < config_d.totalWeights; i += stride)
            {
                rand = curand_uniform(&state) * randomMagnitude * 2 - randomMagnitude;
                (nextGenWeights)[i + offsetBot2 * config_d.totalWeights] = (allWeights)[i + winnerBotOffset * config_d.totalWeights] + rand;
            }

            // We can skip the first layer since the input layer shouldn't have biases.
            for (int i = tid + config_d.layerShapes[0]; i < config_d.totalNeurons; i += stride)
            {
                rand = curand_uniform(&state) * randomMagnitude * 2 - randomMagnitude;
                (nextGenBiases)[i + offsetBot2 * config_d.totalNeurons] = (allBiases)[i + winnerBotOffset * config_d.totalNeurons] + rand;
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


__device__ int counter=0;
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
            // hard coding for TargetSimulation
            if (tid == 0)
            {
                gamestate[0] = 0;
                gamestate[1] = 0;
                gamestate[2] = startingParams[3];
                gamestate[3] = startingParams[4];
                gamestate[4] = startingParams[0];
                gamestate[5] = startingParams[1];
                
                gamestate[6] = 0; //total dist
                
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
                // Set the activations for this iteration


                // hard coding for TargetSimulation:
                const int numInputs = 6;
                if (tid < numInputs)
                {
                    activs[0][tid] = gamestate[tid];
                }
                if(tid == 0){
                    gamestate[7] = iter;

                    //TESTING: not giving velocity as an input:
                    // activs[0][0] = 0;
                    // activs[0][1] = 0;
                }
                // It's important to remember that activs and ws and bs are essentially 2d arrays. That's why indexing them is tricky and weird.
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
                if (iter >= maxIters)
                {
                    finished = true;
                    if(gamestate[6] == 0)
                        output[block] = 0;
                    else if (tid == 0)
                        //output[block] = -gamestate[6]; // Uses totalDist as a metric
                        output[block] = (startingParams[2] / gamestate[6]); // Uses efficiency as a metric
                    
                    if(gid == 0){
                        if(counter % 10 == 0)
                            printf("Block %d total dist = %f, efficiency = %f\n", blockIdx.x, gamestate[6], (startingParams[2] / gamestate[6]));

                        counter++;

                    }
                    
                    // if (gid == 0)
                    // {
                    //     if (tid == 0 && blockIdx.x == 0)
                    //     {
                    //         printf("Activations:\n");
                    //         int AO = 0; // "activs offset"
                    //         for (int layer = 0; layer < config_d.numLayers; layer++)
                    //         {
                    //             printf("Layer %d, size = %d, AO = %d\n", layer, config_d.layerShapes[layer], AO);
                    //             for (int i = 0; i < config_d.layerShapes[layer]; i++)
                    //             {
                    //                 printf("%f, ", activs[0][AO + i]);
                    //             }
                    //             AO += config_d.layerShapes[layer];
                    //             printf("\n");
                    //         }
                    //         printf("\n");
                    //     }
                    // }
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