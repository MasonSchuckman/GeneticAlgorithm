#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include "cuda_runtime.h"
#include "math.h"
#include <math.h>
#include "device_launch_parameters.h"
#include <stdio.h>

//abstract class (interface) for what defines a valid simulation
#pragma once

// These are needed for compile time understanding of static arrays in kernels.
// The code would be much uglier without them.
const int MAX_LAYERS = 6;
const int MAX_BOTS_PER_SIM = 2;

struct SimConfig{
    SimConfig(){}

    SimConfig(int numLayers,
    int totalNeurons,
    int totalWeights,
    int bpb, //Bots per Block (bots per simulation)
    int maxIters,
    int numStartingParams,
    int directContest,
    int botsPerTeam) : numLayers{numLayers}, totalNeurons{totalNeurons}, totalWeights{totalWeights}, bpb{bpb}, 
    maxIters{maxIters}, numStartingParams{numStartingParams}, directContest{directContest}, botsPerTeam{botsPerTeam}{};

    int numLayers;
    int totalNeurons;
    int totalWeights;
    int bpb; //Bots per Block (bots per simulation)
    int maxIters;
    int numStartingParams;

    int directContest; //true if we have 2+ bots competing in a block
    int botsPerTeam; //only relavent if directContest=1. How many bots are on each team. (Assumed 2 teams per block if direct contest = 1)

    // Making this statically allocated might have unforseen consequences, idk.
    int layerShapes[MAX_LAYERS];
    int layerTypes[MAX_LAYERS];
    int paddedNetworkSize;
};

class Simulation {
public:
    __host__ __device__ Simulation() {}
    __host__ __device__ virtual ~Simulation() {}

    __host__ virtual void getStartingParams(float * startingParams, int tid, int blockid) = 0;
    // actions and gamestate are both shared memory variables. The exact way they're used is
    // simulation dependent.
    __device__ virtual void eval(float ** actions, float * gamestate) = 0;

    //Called at the beginning of the kernel. Used to do things like place the bots at their starting positions and such
    __device__ virtual void setupSimulation(const float * startingParams, float * gamestate) = 0;

    //Called at the beginning of each sim iteration. 
    __device__ virtual void setActivations(float * gamestate, float ** activs, int iter) = 0;


    //return 1 if simulation is finished.
    __device__ virtual int checkFinished(float * gamestate) = 0;
    //__device__ virtual void determineOutcome(float * gamestate) = 0; //TODO: determine if this is needed

    __device__ virtual void setOutput(float * output, float * gamestate, const float * startingParams_d) = 0;

    //NOTE: The ID this function returns MUST be unique for each derived class!
    __host__ virtual int getID() = 0;
    
    SimConfig* config;

};

// CPU version of sim config. Contains EVERYTHING that could go into a json file.
struct FullSimConfig{
    Simulation * sim;
    SimConfig config;

    int totalBots;
    int generations;
    float baseMutationRate;
    float minMutationRate;
    float mutationDecayRate;
    float shiftEffectiveness;

    int loadData;
    FullSimConfig(Simulation * sim_, SimConfig config_, int totalBots_, int generations_, float baseMutationRate_,
    float minMutationRate_, float mutationDecayRate_, float shiftEffectiveness_, int loadData_)
        : sim(sim_), config(config_), totalBots(totalBots_), generations(generations_), baseMutationRate(baseMutationRate_)
        , minMutationRate(minMutationRate_), mutationDecayRate(mutationDecayRate_), shiftEffectiveness(shiftEffectiveness_), loadData(loadData_) {}

    ~FullSimConfig(){
        delete sim;
    }
};




#endif
