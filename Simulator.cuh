#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include "cuda_runtime.h"
#include "math.h"
#include <math.h>
#include "device_launch_parameters.h"
#include <stdio.h>

#include "Bot.h"
#include "Simulation.cuh"
#include "BasicSimulation.cuh"
#include "Kernels.cuh"

#include <iostream>
#include <vector>
using std::vector;

#define check(ans)                            \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU error check: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}


class Simulator
{
public:
    Simulator(){}
    // Constructor allocates all necessary device memory prior to doing simulations
    Simulator(vector<Bot*> bots, Simulation* derived, SimConfig &config);

    ~Simulator();

    void simulate();

    void batchSimulate(int numSimulations);

    Bot * getBest();

private:

    void formatBotData(int *& layerShapes_h, float *&startingParams_h, 
        float *&output_h, float *&weights_h, float *&biases_h);

    void copyToGPU(int *& layerShapes_h, float *&startingParams_h, 
        float *&output_h, float *&weights_h, float *&biases_h);

    void copyFromGPU(float *&weights_h, float *&biases_h);


    void runSimulation(float * output_h);
    
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

    float* nextGenWeights_d;
    float* nextGenBiases_d;

    vector<Bot*> bots;
    Simulation * derived;
    Simulation **sim_d;

    SimConfig config;

};

#endif