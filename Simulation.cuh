#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include "cuda_runtime.h"
#include "math.h"
#include <math.h>
#include "device_launch_parameters.h"
#include <stdio.h>

//abstract class (interface) for what defines a valid simulation
#pragma once

const int MAX_LAYERS = 20;

// Max layers = 20 right now.
struct SimConfig{
    int numLayers;
    int totalNeurons;
    int totalWeights;
    int bpb; //Bots per Block (bots per simulation)
    int maxIters;

    
    // Making this statically allocated might have unforseen consequences, idk.
    int layerShapes[MAX_LAYERS];
    
};

class Simulation {
public:
    __host__ __device__ Simulation() {}
    __host__ __device__ virtual ~Simulation() {}

    // actions and gamestate are both shared memory variables. The exact way they're used is
    // simulation dependent.
    __device__ virtual void eval(float * actions, float * gamestate) = 0;

    //return 1 if simulation is finished.
    __device__ virtual int checkFinished(float * gamestate) = 0;
    //__device__ virtual void determineOutcome(float * gamestate) = 0; //TODO: determine if this is needed


    //NOTE: The ID this function returns MUST be unique for each derived class!
    __host__ virtual int getID() = 0;
    
    SimConfig* config;

};

#endif
