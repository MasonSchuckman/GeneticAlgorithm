#include "Simulation.cuh"
#include "BasicSimulation.cuh"

namespace Kernels
{
    __global__ void game_kernel(int n, Simulation ** sim);

    __global__ void createDerived(Simulation **sim, int id);

    __global__ void delete_function(Simulation **sim);

    __global__ void simulateShared(const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

};