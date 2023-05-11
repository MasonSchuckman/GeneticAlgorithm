#include "SimulationList.h"
#include <math.h>
#include <random>

extern SimConfig config_d;

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
    void forward_propagation(int tid, int blockIdx, const float *inputs, const float *weights, const float *biases, float *output, int input_size, int output_size, int layer)
    {
        int stride = 32;

        // Initialize output to biases
        for (int i = 0; i < output_size; i++)
        {
            output[i] = biases[i];
        }

        // Compute dot product of input and weights
        for (int i = 0; i < input_size; i++)
        {
            for (int j = 0; j < output_size; j++)
            {
                output[j] += inputs[i] * weights[i * output_size + j];
            }
        }

        //  Apply activation function
        switch (config_d.layerTypes[layer])
        {
        // linear
        case 0:
            break;

        // ReLU
        case 1:
        {
            for (int i = 0; i < output_size; i++)
                output[i] = output[i] > 0 ? output[i] : 0; // max(output[i],0)
        }
        break;

        // Sigmoid
        case 2:
        {
            for (int i = 0; i < output_size; i++)
                output[i] = 1.0f / (1.0f + expf(-output[i]));
        }
        break;

        // Default is linear
        default:
            break;
        }
    }

    // Each block will go through each layer of its respective bot(s), and threads will edit individual weights/biases.
    // The nextGenWeights/biases arrays are the exact same shape and size of the allWeights/biases arrays, but with the genetic information of the next generation.
    void mutate(int block, const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome, int *childSpecies,
                float *nextGenWeights, float *nextGenBiases, float *distances, float *deltas, int *ancestors, float progThreshold, const int gen, const int shift)
    {

        int stride = 32;

        // prevent OOB errors
        if (block < n / 2)
        {

            float rand = 0;
            // calcuate the offset for this block's bot(s)
            int offsetBot1 = block * 2;
            int offsetBot2 = (block * 2 + shift * 2 + 1) % n;
            int inputBotOffsets[2] = {offsetBot1, offsetBot2};
            int outputBotOffsets[2] = {offsetBot1, offsetBot2};

            // Compare the two bots that competed in the mutation phase
            if (config_d.directContest == 1)
            {
                inputBotOffsets[0] = offsetBot1;
                inputBotOffsets[1] = offsetBot1 + 1;
            }

            float botScore1 = simulationOutcome[inputBotOffsets[0]];
            float botScore2 = simulationOutcome[inputBotOffsets[1]];
            if (botScore1 == 0 && botScore2 == 0)
                printf("Error. Both zero. block = %d, offset1 = %d, offset2 = %d\n", block, inputBotOffsets[0], inputBotOffsets[1]);

            int winnerBotOffset;
            if (botScore1 > botScore2)
            {
                winnerBotOffset = inputBotOffsets[0];
            }
            else
            {
                winnerBotOffset = inputBotOffsets[1];
            }

            // keeping track of the parent specimen from which the children came from
            childSpecies[outputBotOffsets[0]] = winnerBotOffset;
            childSpecies[outputBotOffsets[1]] = winnerBotOffset;

            float biasMagnification = 10.0f;

            float distance = 0;       // Distance from the parent
            float deltaMagnitude = 0; // deltaMagnitude is the L1 norm of a bot's genome and the progenitor it decended from.

            for (int bot = 0; bot < 2; bot++)
            {
                // Write this bot's updated weights
                for (int i = 0; i < config_d.totalWeights; i++)
                {
                    if (bot == 1) // Only add noise to bot 1 (bot 0 stays the same as the winner)
                        rand = ((double)std::rand() / (RAND_MAX)) * randomMagnitude * 2 - randomMagnitude;
                    distance += std::abs(rand);
                    //printf("INFO : %d %d\n", i + outputBotOffsets[bot] * config_d.paddedNetworkSize, config_d.paddedNetworkSize);
                    (deltas)[i + outputBotOffsets[bot] * config_d.paddedNetworkSize] += rand;
                    (nextGenWeights)[i + outputBotOffsets[bot] * config_d.totalWeights] = (allWeights)[i + winnerBotOffset * config_d.totalWeights] + rand;
                }

                // Write this bot's updated biases
                // We can skip the first layer since the input layer shouldn't have biases.
                for (int i = 0 + config_d.layerShapes[0]; i < config_d.totalNeurons; i++)
                {
                    if (bot == 1) // Only add noise to bot 1 (bot 0 stays the same as the winner)
                        rand = ((double)std::rand() / (RAND_MAX)) * randomMagnitude * biasMagnification * 2 - randomMagnitude * biasMagnification;
                    distance += std::abs(rand);

                    (deltas)[i + config_d.totalWeights + outputBotOffsets[bot] * config_d.paddedNetworkSize] += rand;
                    (nextGenBiases)[i + outputBotOffsets[bot] * config_d.totalNeurons] = (allBiases)[i + winnerBotOffset * config_d.totalNeurons] + rand;
                }

                float totalDistance = distance;
                // deltaMagnitude = block_reduce<float>(&(deltas[outputBotOffsets[bot] * config_d.paddedNetworkSize]), config_d.paddedNetworkSize);
                for (int i = 0; i < config_d.paddedNetworkSize; i++)
                {
                    deltaMagnitude += std::abs(deltas[outputBotOffsets[bot] + i]);
                }

                (distances)[outputBotOffsets[bot]] = totalDistance;
                //if (block == 0)
                //    printf("delta mag = %f\n", deltaMagnitude);
                // Check if child is a new species
                if (deltaMagnitude >= progThreshold)
                {
                    (ancestors)[outputBotOffsets[bot]] = outputBotOffsets[bot] + gen * n;

                    // Reset the deltas for this bot since it is now the prog
                    // zeroArray(&(deltas[outputBotOffsets[bot] * config_d.paddedNetworkSize]), config_d.paddedNetworkSize);
                    for (int i = 0; i < config_d.paddedNetworkSize; i++)
                    {
                        deltas[outputBotOffsets[bot] + i] = 0;
                    }
                }
            }
        }

        return;
    }

    // this kernel divides the work into blocks rather than each thread works alone.
    // Reason for doing this is to hopefully make better use of cache and reduce memory stalls.
    // (Hopefully this will lead to higher FLOPS).

    int counter = 0;
    void simulateShared2(int blockIdx, float *s, const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output)
    {
        // int gid = tid + blockIdx * 32; // global id

        int block = blockIdx;
        int stride = 32;

        // prevent OOB errors
        if (block < n)
        {

            // hard coding this makes things *much* simpler. We can change it if needed.
            float gamestate[64];

            (*sim)->setupSimulation(0, blockIdx, startingParams, gamestate);

            // shared mem layout is w1,w2...,w_bpt,b1,b2,...,b_bpt,a_1,a_2,...,a_bpt
            // declare our block of shared memory
            // float s[];

            // split our shared memory block into the weights, biases, and activations
            float *weights = s;
            float *biases = weights + config_d.totalWeights * config_d.bpb;
            float *activations = biases + config_d.totalNeurons * config_d.bpb;

            // Copy this block's weights and biases to the shared arrays.
            for (int i = 0; i < config_d.totalWeights * config_d.bpb; i++)
            {
                weights[i] = (allWeights)[block * config_d.totalWeights * config_d.bpb + i];
            }
            for (int i = 0; i < config_d.totalNeurons * config_d.bpb; i++)
            {
                biases[i] = (allBiases)[block * config_d.totalNeurons * config_d.bpb + i];
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

            int maxIters = config_d.maxIters;
            bool finished = false;

            int iter = 0; // current timestep of simulation we're on

            // run the simulation loop.
            while (!finished)
            {
                // Set the activations for this bot this iteration
                (*sim)->setActivations(0, block, gamestate, activs, iter);

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
                        forward_propagation(0, block, activs[bot] + AO, ws[bot] + WO, bs[bot] + numBiases + BO, activs[bot] + AO + numBiases, numBiases, config_d.layerShapes[layer + 1], layer);

                        // This register-less version had almost identical performance
                        // forward_propagation(activs(bot) + numBiases * layer, ws(bot) + numWeights * layer, bs(bot) + numBiases * (layer + 1), activs(bot) + numBiases * (layer + 1), numBiases, config_d.layerShapes[layer + 1]);

                        AO += numBiases;
                        WO += numWeights;
                        BO += numBiases;
                    }
                }

                // update simulation/game state based on bot actions
                (*sim)->eval(0, block, actions, gamestate);

                if ((*sim)->checkFinished(0, block, gamestate) == 1)
                {
                    finished = true;
                }

                iter++;
                if (iter >= maxIters || finished)
                {
                    finished = true;
                    (*sim)->setOutput(0, block, output, gamestate, startingParams);
                }
            }
        }

        return;
    }
}