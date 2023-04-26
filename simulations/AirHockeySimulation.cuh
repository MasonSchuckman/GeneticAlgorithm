#ifndef AIRHOCKEY_SIMULATION_CUH
#define AIRHOCKEY_SIMULATION_CUH

#pragma once

#include "Simulation.cuh"


class AirHockeySimulation : public Simulation {
public:
    __host__ __device__ MultibotSimulation(){}    
    __host__ __device__ ~MultibotSimulation(){}

    __host__ void getStartingParams(float * startingParams);

    //Called at the beginning of the kernel. Used to do things like place the bots at their starting positions and such
    __device__ void setupSimulation(const float * startingParams, float * gamestate);

    //Called at the beginning of each sim iteration. 
    __device__ void setActivations(float * gamestate, float ** activs, int iter);

    __device__ void eval(float ** actions, float * gamestate);

    __device__ int checkFinished(float * gamestate);

    __device__ void setOutput(float * output, float * gamestate, const float * startingParams_d);

    
    __host__ int getID();

    
};

#endif