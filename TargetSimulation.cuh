#ifndef TARGET_SIMULATION_CUH
#define TARGET_SIMULATION_CUH

#pragma once

#include "Simulation.cuh"


class TargetSimulation : public Simulation {
public:
    __host__ __device__ TargetSimulation(){}    
    __host__ __device__ ~TargetSimulation(){}

    
    __device__ void eval(float ** actions, float * gamestate);

    __device__ int checkFinished(float * gamestate);

    __host__ int getID();

    
};

#endif