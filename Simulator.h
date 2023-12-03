#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "math.h"
#include <math.h>
#include <stdio.h>

#include "Bot.h"
#include "biology/Taxonomy.h"
#include "biology/Specimen.h"
#include "SimulationList.h"
#include "Kernels.h"
#include "Agent.h"

#include <iostream>
#include <vector>
using std::vector;




class Simulator
{
public:
    Simulator(){}
    // Constructor allocates all necessary device memory prior to doing simulations
    Simulator(vector<Specimen*> bots, Simulation* derived, SimConfig &config, Taxonomy *history);

    ~Simulator();

    void simulate();

    

    void batchSimulate(int numSimulations);

    Bot * getBest();

    float mutateMagnitude = 1.0f; //starting magnitude
    float min_mutate_rate = .000001f; //ending magnitude
    float mutateDecayRate = 0.99f;
    float shiftEffectiveness = 1.0f;
    int loadData = 0;
    int RL = 0;
    int NUM_THREADS = 4; // Number of threads

private:
    // void processBlocksSimulate(int startBlock, int endBlock, int sharedMemNeeded, int numBlocks,
    //                const float* weights_d, const float* biases_d, const float* startingParams_d, float* output_d);
    

    // void processBlocksMutate(int startBlock, int endBlock, int totalBots, float mutateMagnitude, float* weights_d,
    //                float* biases_d, float* output_d, int* parentSpecimen_d, float* nextGenWeights_d,
    //                float* nextGenBiases_d, float* distances_d, float* deltas_d, int* ancestors_d,
    //                float progThreshold, int iterationsCompleted, int shift);

    float getAvgDistance();

    void formatBotData(int *& layerShapes_h, float *&startingParams_h, 
        float *&output_h, float *&weights_h, float *&biases_h);

    void copyToGPU(int *& layerShapes_h, float *&startingParams_h, 
        float *&output_h, float *&weights_h, float *&biases_h);

    void copyFromGPU(float *&weights_h, float *&biases_h);


    //void runSimulation(float *output_h, int *parentSpecimen_h, int* ancestors_h, float* distances_h);
    std::vector<episodeHistory> runSimulation(float *output_h, int *parentSpecimen_h, int* ancestors_h, float* distances_h);

    std::vector<episodeHistory> runSimulationRL(Agent & agent, float *output_h);


    void readWeightsAndBiasesAll(float *&weights_h, float *&biases_h, int &TOTAL_BOTS, int &totalWeights, int &totalNeurons, int &numLayers, int * layerShapes);

    //Reads in saved weights and biases if it matches the current config
    void loadData_(float *weights_h, float *biases_h);

    int iterationsCompleted = 0;

    
    /*
    Pointers to device memory should be private and persist as we're doing simulation.
    This way we don't need to constantly re allocate device memory.
    */
    int *layerShapes_d;
    float *startingParams_d;
    float *output_d;
    float *weights_d;
    float *biases_d;
    int *parentSpecimen_d;
    float *distances_d;
    float * deltas_d;
    
    float* nextGenWeights_d;
    float* nextGenBiases_d;
    int * ancestors_d;

    vector<Specimen*> bots;
    Taxonomy *history;
    Simulation * derived;
    Simulation **sim_d;

    SimConfig config;

};

#endif