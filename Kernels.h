#ifndef KERNELS_H
#define KERNELS_H
#include "SimulationList.h"


namespace Kernels
{
    void simulateShared2(int block, float *s, const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

    void mutate(int block, const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome, int *childSpecies, 
                           float *nextGenWeights, float *nextGenBiases, float * distances,  float * deltas, int * ancestors, float progThreshold, const int gen, const int shift);
};

#endif