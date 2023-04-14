#ifndef BASIC_SIMULATION_CUH
#define BASIC_SIMULATION_CUH

#pragma once

#include "Simulation.cuh"

class BasicSimulation : public Simulation {
public:
    __host__ __device__ BasicSimulation(){}    
    __host__ __device__ ~BasicSimulation(){}

    
    __device__ void eval(float ** actions, float * gamestate);

    __device__ int checkFinished(float * gamestate);

    __host__ int getID();

    
};

#endif