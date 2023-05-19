#ifndef KERNELS_CUH
#define KERNELS_CUH
#include "SimulationList.cuh"


namespace Kernels
{
    __global__ void game_kernel(int n, Simulation ** sim);

    __global__ void createDerived(Simulation **sim, int id);

    __global__ void delete_function(Simulation **sim);

    __global__ void simulateShared(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

    __global__ void simulateShared_noStaticArrays(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

    __global__ void simulateShared2(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

    __global__ void simulateShared3(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

    __global__ void mutate(const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome, int *childSpecies, 
                           float *nextGenWeights, float *nextGenBiases, float * distances,  float * deltas, int * ancestors, float progThreshold, const int gen, const int shift);
};

#endif