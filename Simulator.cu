#include "Simulator.cuh"
#include <random>
#include <cmath>

using std::vector;

extern __constant__ SimConfig config_d;

// Constructor allocates all necessary device memory prior to doing simulations
Simulator::Simulator(vector<Bot *> bots, Simulation *derived, SimConfig &config) : bots{bots}, config{config}, derived{derived}
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
    for (int i = 0; i < config.numLayers; i++)
    {
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
                    biases_h[i * config.totalNeurons + BO + k] = 0;
                }
                if (j != config.numLayers - 1)
                {
                    for (int l = 0; l < layerShapes_h[j + 1]; l++)
                    {
                        // set the weights. all layers get a weight of layerNum+1
                        weights_h[i * config.totalNeurons + WO + k * layerShapes_h[j + 1] + l] = 0;
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

    for (int i = 0; i < totalBots * config.totalNeurons; i++)
        biases_h[i] = 0;
    for (int i = 0; i < totalBots * config.totalWeights; i++)
        weights_h[i] = 0;
}

void Simulator::copyToGPU(int *&layerShapes_h, float *&startingParams_h,
                          float *&output_h, float *&weights_h, float *&biases_h)
{
    int totalBots = bots.size();
    check(cudaMemcpy(layerShapes_d, layerShapes_h, config.numLayers * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(startingParams_d, startingParams_h, config.numStartingParams * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output_d, output_h, totalBots * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(weights_d, weights_h, totalBots * config.totalWeights * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(nextGenWeights_d, weights_h, totalBots * config.totalWeights * sizeof(float), cudaMemcpyHostToDevice));

    check(cudaMemcpy(biases_d, biases_h, totalBots * config.totalNeurons * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(nextGenBiases_d, biases_h, totalBots * config.totalNeurons * sizeof(float), cudaMemcpyHostToDevice));
}

// Copies the weights and biases of all the bots back to the host
void Simulator::copyFromGPU(float *&weights_h, float *&biases_h)
{

    int totalBots = bots.size();
    check(cudaMemcpy(weights_h, nextGenWeights_d, totalBots * config.totalWeights * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(biases_h, nextGenBiases_d, totalBots * config.totalNeurons * sizeof(float), cudaMemcpyDeviceToHost));
}

#include <fstream>
void writeWeightsAndBiasesAll(float *weights_h, float *biases_h, int TOTAL_BOTS, int totalWeights, int totalNeurons, int numLayers, int *layerShapes) {
    std::ofstream outfile("allBots.data", std::ios::out | std::ios::binary); //this might be more space efficient
    //std::ofstream outfile("allBots.data");
    // outfile << "all bots:\n";
    // Write the total number of bots
    outfile.write(reinterpret_cast<const char*>(&TOTAL_BOTS), sizeof(int));

    // Write the total number of weights and neurons
    outfile.write(reinterpret_cast<const char*>(&totalWeights), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&totalNeurons), sizeof(int));

    // Write the number of layers and their shapes
    outfile.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));
    for (int i = 0; i < numLayers; i++) {
        outfile.write(reinterpret_cast<const char*>(&layerShapes[i]), sizeof(int));
    }

    // Write the weights and biases for each bot
    for (int bot = 0; bot < TOTAL_BOTS; bot++) {
        // Write the weights for this bot
        for (int i = 0; i < totalWeights; i++) {
            float weight = weights_h[bot * totalWeights + i];
            outfile.write(reinterpret_cast<const char*>(&weight), sizeof(float));
        }
        

        // Write the biases for this bot
        int biasOffset = bot * totalNeurons;
        for (int i = 0; i < totalNeurons; i++) {
            float bias = biases_h[biasOffset + i];
            outfile.write(reinterpret_cast<const char*>(&bias), sizeof(float));
        }
        
    }

    outfile.close();
}

void write_weights_and_biases(float* weights, float* biases, int numLayers, int* layerShapes, int totalWeights, int totalNeurons, int lastGenBest) {
    std::ofstream outfile("bestBot.data");
    outfile << "net_weights = np.array([";
    int WO = 0;
    for (int layer = 0; layer < numLayers - 1; layer++)
    {
        int numWeightsInLayer = layerShapes[layer] * layerShapes[layer + 1];
        outfile << "[";
        for (int i = 0; i < numWeightsInLayer; i++)
        {
            outfile << weights[lastGenBest * totalWeights + WO + i];
            if (i != numWeightsInLayer - 1)
                outfile << ", ";
        }
        WO += numWeightsInLayer;
        outfile << "]";
        if (layer != numLayers - 2)
            outfile << ",\n";
    }
    outfile << "])\n";

    int BO = layerShapes[0];
    outfile << "net_biases = np.array([";
    for (int layer = 1; layer < numLayers; layer++)
    {
        outfile << "[";
        for (int i = 0; i < layerShapes[layer]; i++)
        {
            outfile << biases[lastGenBest * totalNeurons + BO + i];
            if (i != layerShapes[layer] - 1)
                outfile << ", ";
        }
        BO += layerShapes[layer];
        outfile << "]";
        if (layer != numLayers - 1)
            outfile << ",\n";
    }
    outfile << "])\n";
    outfile.close();
}

void printError(){
    printf("Error in loadData_()! Saved config doesn't match current config. Turn off load_data in the json.\n");
    exit(1);
}

// Dumb load. Assumes load will work (same number of bots and network config)
void Simulator::loadData_(float *weights_h, float *biases_h){
    std::ifstream infile("allBots.data", std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file\n";
        exit(1);
    }
    int placeholder;
    // Read the total number of bots
    infile.read(reinterpret_cast<char*>(&placeholder), sizeof(int));
    if(placeholder != bots.size()){
        printError();
    }

    // Read the total number of weights and neurons
    infile.read(reinterpret_cast<char*>(&placeholder), sizeof(int));
    if(placeholder != config.totalWeights){
        printError();
    }
    infile.read(reinterpret_cast<char*>(&placeholder), sizeof(int));
    
    // Read the number of layers and their shapes
    infile.read(reinterpret_cast<char*>(&placeholder), sizeof(int));
        
    int TOTAL_BOTS = bots.size();
    int totalWeights = config.totalWeights;
    int totalNeurons = config.totalNeurons;

    // Read the weights and biases for each bot
    for (int bot = 0; bot < TOTAL_BOTS; bot++) {
        // Read the weights for each layer
        for (int i = 0; i < totalWeights; i++) {
            float weight;
            infile.read(reinterpret_cast<char*>(&weight), sizeof(float));
            weights_h[bot * totalWeights + i] = weight;
        }
        
        // Read the biases for each layer
        for (int i = 0; i < totalNeurons; i++) {
            float bias;
            infile.read(reinterpret_cast<char*>(&bias), sizeof(float));
            biases_h[bot * totalNeurons + i] = bias;
        }
    }

    infile.close();
}

#include <sstream>
void Simulator::readWeightsAndBiasesAll(float *&weights_h, float *&biases_h, int &TOTAL_BOTS, int &totalWeights, int &totalNeurons, int &numLayers, int * layerShapes) {
    std::ifstream infile("allBots.data", std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file\n";
        exit(1);
    }

    // Read the total number of bots
    infile.read(reinterpret_cast<char*>(&TOTAL_BOTS), sizeof(int));

    // Read the total number of weights and neurons
    infile.read(reinterpret_cast<char*>(&totalWeights), sizeof(int));
    infile.read(reinterpret_cast<char*>(&totalNeurons), sizeof(int));

    // Read the number of layers and their shapes
    infile.read(reinterpret_cast<char*>(&numLayers), sizeof(int));
    layerShapes = new int[numLayers];
    for (int i = 0; i < numLayers; i++) {
        infile.read(reinterpret_cast<char*>(&layerShapes[i]), sizeof(int));
    }

    // Allocate memory for the weights and biases
    weights_h = new float[TOTAL_BOTS * totalWeights];
    biases_h = new float[TOTAL_BOTS * totalNeurons];

    // Read the weights and biases for each bot
    for (int bot = 0; bot < TOTAL_BOTS; bot++) {
        // Read the weights for each layer
        for (int i = 0; i < totalWeights; i++) {
            float weight;
            infile.read(reinterpret_cast<char*>(&weight), sizeof(float));
            weights_h[bot * totalWeights + i] = weight;
        }
        
        // Read the biases for each layer
        for (int i = 0; i < totalNeurons; i++) {
            float bias;
            infile.read(reinterpret_cast<char*>(&bias), sizeof(float));
            biases_h[bot * totalNeurons + i] = bias;
        }
    }

    infile.close();
}

void read_weights_and_biases(float* weights, float* biases, int numLayers, int* layerShapes, int totalWeights, int totalNeurons, int lastGenBest) {
    std::ifstream infile("bestBot.data");
    std::string line;
    std::vector<float> weights_vec;
    std::vector<float> biases_vec;
    bool reading_weights = false;
    bool reading_biases = false;
    int WO = 0;
    int BO = layerShapes[0];
    int layer = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        while (std::getline(iss, token, '[')) {
            if (token.find("net_weights") != std::string::npos) {
                reading_weights = true;
                continue;
            }
            else if (token.find("net_biases") != std::string::npos) {
                reading_biases = true;
                continue;
            }
            if (!reading_weights && !reading_biases) {
                continue;
            }
            else if (reading_weights && token.find("]") != std::string::npos) {
                reading_weights = false;
                layer++;
            }
            else if (reading_biases && token.find("]") != std::string::npos) {
                reading_biases = false;
                layer++;
            }
            else {
                std::istringstream inner_iss(token);
                std::string inner_token;

                while (std::getline(inner_iss, inner_token, ',')) {
                    if (reading_weights) {
                        weights_vec.push_back(std::stof(inner_token));
                    }
                    else if (reading_biases) {
                        biases_vec.push_back(std::stof(inner_token));
                    }
                }
            }
        }
    }

    int weight_index = lastGenBest * totalWeights;
    int bias_index = lastGenBest * totalNeurons;
    WO = 0;
    BO = layerShapes[0];

    for (int layer = 0; layer < numLayers - 1; layer++) {
        int numWeightsInLayer = layerShapes[layer] * layerShapes[layer + 1];

        for (int i = 0; i < numWeightsInLayer; i++) {
            weights[weight_index + WO + i] = weights_vec[WO + i];
        }
        WO += numWeightsInLayer;
    }

    for (int layer = 1; layer < numLayers; layer++) {
        int numBiasesInLayer = layerShapes[layer];

        for (int i = 0; i < numBiasesInLayer; i++) {
            biases[bias_index + BO + i] = biases_vec[BO + i];
        }
        BO += numBiasesInLayer;
    }

    infile.close();
}


#include <chrono>


void Simulator::runSimulation(float *output_h)
{
    int totalBots = bots.size();
    int tpb = 32; // threads per block
    int numBlocks = (totalBots / config.bpb);

    int sharedMemNeeded = (config.totalWeights + config.totalNeurons * 2) * config.bpb;
    if(iterationsCompleted == 0){
        printf("Num blocks = %d. Bots per sim = %d\n", numBlocks, config.bpb);
        printf("Shared mem needed per block = %d KB\n", sharedMemNeeded * sizeof(float) / (2 << 10));
    }
    
    float * startingParams_h = new float[config.numStartingParams];
    derived->getStartingParams(startingParams_h);

    check(cudaMemcpy(startingParams_d, startingParams_h, config.numStartingParams * sizeof(float), cudaMemcpyHostToDevice));
    delete[] startingParams_h;

    auto start_time = std::chrono::high_resolution_clock::now();
    // Launch a kernel on the GPU with one block for each simulation/contest
    Kernels::simulateShared2<<<numBlocks, tpb, sharedMemNeeded * sizeof(float)>>>(numBlocks, this->sim_d, weights_d, biases_d, startingParams_d, output_d);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    check(cudaDeviceSynchronize());
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // std::cout << "Simulation time taken: " << elapsed_time << " ms\n";

    // Idea for mutating:
    /*
    if not a direct contest:
        One block looks at two bots, determined by the blockId * 2, and (blockId * 2 + iter*2 + 1 + (optional) shuffleSeed*2) % numBots
        The block can then write the 2 new children data where the parents were.

    if a direct contest:
        if 1 bot per team:
            each block looks at two bots, and checks which one won. The winner will split into two children, and the block will write the data to
            blockId * 2 and (blockId*2 + iter*2 + 1 + (optional) shuffleSeed*2) % numBots (assumes numBots is even.)
        if 2 bots per team:


    */

    
    // slowly reduce the mutation rate until it hits a lower bound
    if (mutateMagnitude > min_mutate_rate)
        mutateMagnitude *= mutateDecayRate;

    // each block looks at 2 bots
    numBlocks = totalBots / 2; //(assumes even number of bots)
    //start_time = std::chrono::high_resolution_clock::now();
    
    int shift = (int) (((double)rand() / RAND_MAX) * totalBots * shiftEffectiveness) % totalBots;
    if(shiftEffectiveness < 0)
        shift = iterationsCompleted;
    Kernels::mutate<<<numBlocks, tpb>>>(totalBots, mutateMagnitude, weights_d, biases_d, output_d, nextGenWeights_d, nextGenBiases_d, shift);
    check(cudaDeviceSynchronize());
    end_time = std::chrono::high_resolution_clock::now();

    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    // std::cout << "Mutation time taken: " << elapsed_time << " ms\n";

    // swap which weights/biases arrays are "current"
    float *temp = nextGenBiases_d;
    nextGenBiases_d = biases_d;
    biases_d = temp;

    temp = nextGenWeights_d;
    nextGenWeights_d = weights_d;
    weights_d = temp;

    // Copy output vector from GPU buffer to host memory.
    check(cudaMemcpy(output_h, output_d, totalBots * sizeof(float), cudaMemcpyDeviceToHost));

    // Used to decide where to write nextGen population data to
    iterationsCompleted++;
    if (iterationsCompleted % 25 == 0){
        printf("iter %d, mutate scale = %f. Shift = %d", iterationsCompleted, mutateMagnitude, shift);
        std::cout << " Generation took " << elapsed_time << " ms.\n";
    }
}

void analyzeHistory(int numSimulations, int totalBots, float *output_h, int &finalBest)
{

    int printInterval = 25;
    int *bestIndexes = new int[numSimulations];
    float *bestScores = new float[numSimulations];
    float *averageScores = new float[numSimulations];

    // Loop over every generation
    for (int i = 0; i < numSimulations; i++)
    {
        float bestScore = -1000000;
        float averageScore = 0;
        int bestIndex = -1;

        // Find the best scoring bot of this generation
        for (int bot = 0; bot < totalBots; bot++)
        {
            float score = output_h[i * totalBots + bot];
            averageScore += score;
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = bot;
            }
        }

        // Record the results of this iteration
        bestIndexes[i] = bestIndex;
        bestScores[i] = bestScore;
        averageScores[i] = averageScore / totalBots;
    }

    // Summarize the results
    for (int i = 0; i < numSimulations; i += printInterval)
    {
        printf("Iteration : [%d]\tTop Score : %f, by Bot : [%d]\tAverage score : %f\n", i, bestScores[i], bestIndexes[i], averageScores[i]);
    }

    finalBest = bestIndexes[numSimulations - 1];
    //finalBest = 0;

    delete[] bestIndexes;
    delete[] bestScores;
    delete[] averageScores;
}

void Simulator::batchSimulate(int numSimulations)
{

    printf("num bots = %d, numLayers = %d, num weights = %d, numNeurons = %d\n", bots.size(), config.numLayers, config.totalWeights, config.totalNeurons);
    int totalBots = bots.size();

    // Allocate storage for bot data
    int *layerShapes_h = new int[config.numLayers];
    float *startingParams_h = new float[config.numStartingParams];
    float *output_h = new float[totalBots * numSimulations]; // We'll record all scores for all generations.
    float *weights_h = new float[config.totalWeights * totalBots];
    float *biases_h = new float[config.totalNeurons * totalBots];
    printf("Allocated host memory.\n");

    // Convert all the bot data to the format we need to transfer to GPU
    formatBotData(layerShapes_h, startingParams_h, output_h, weights_h, biases_h);
    printf("Formatted bot data.\n");

    if(loadData == 1){
        loadData_(weights_h, biases_h);
        printf("Loaded in saved weights and biases.\n");
    }
    // Copy it over to the GPU
    copyToGPU(layerShapes_h, startingParams_h, output_h, weights_h, biases_h);
    printf("Copied data to GPU.\n");

    // Invoke the kernel
    for (int i = 0; i < numSimulations; i++)
    {
        // Only pass the location to where this iteration is writing
        runSimulation(&output_h[i * totalBots]);
    }
    printf("Ran simulation.\n");

    copyFromGPU(weights_h, biases_h);

    // Find the best score in each generation
    int lastGenBest = 0;
    analyzeHistory(numSimulations, totalBots, output_h, lastGenBest);

    // Print the last gen's best bot's weights and biases
    printf("net_weights = np.array([");
    int WO = 0;
    for (int layer = 0; layer < config.numLayers - 1; layer++)
    {
        int numWeightsInLayer = config.layerShapes[layer] * config.layerShapes[layer + 1];
        printf("[");
        for (int i = 0; i < numWeightsInLayer; i++)
        {
            printf("%f", weights_h[lastGenBest * config.totalWeights + WO + i]);
            if (i != numWeightsInLayer - 1)
                printf(", ");
        }
        WO += numWeightsInLayer;
        printf("]");
        if (layer != config.numLayers - 2)
            printf(",\n");
    }
    printf("])\n");


    printf("net_biases = np.array([");
    int BO = config.layerShapes[0];
    for (int layer = 1; layer < config.numLayers; layer++)
    {
        printf("[");
        for (int i = 0; i < config.layerShapes[layer]; i++)
        {
            printf("%f", biases_h[lastGenBest * config.totalNeurons + BO + i]);
            if (i != config.layerShapes[layer] - 1)
                printf(", ");
        }
        BO += config.layerShapes[layer];
        printf("]");
        if (layer != config.numLayers - 1)
            printf(",\n");
    }
    printf("])\n");
    

    write_weights_and_biases(weights_h, biases_h, config.numLayers, config.layerShapes, config.totalWeights, config.totalNeurons, lastGenBest);
    writeWeightsAndBiasesAll(weights_h, biases_h, totalBots, config.totalWeights, config.totalNeurons, config.numLayers, config.layerShapes);
    
    float * savedWeights;
    float * savedBiases;

    readWeightsAndBiasesAll(savedWeights, savedBiases, totalBots, config.totalWeights, config.totalNeurons, config.numLayers, config.layerShapes);

    int passed = 1;
    for(int i = 0; i < config.totalWeights * totalBots; i++){
        if(savedWeights[i] != weights_h[i]){
            printf("iter %d\tsaved : %f\ttrue : %f\n",i , savedWeights[i], weights_h[i]);
            passed = 0;
        }
    }
    printf("PASSED TEST? %d\n", passed);

    delete [] savedWeights;
    delete [] savedBiases;
    


    // Do something with the output data....

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
