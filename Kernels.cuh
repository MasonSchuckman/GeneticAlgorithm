#include "Simulation.cuh"
#include "BasicSimulation.cuh"

namespace Kernels
{
    __global__ void game_kernel(int n, Simulation ** sim);

    __global__ void createDerived(Simulation **sim, int id);

    __global__ void delete_function(Simulation **sim);
};