#ifndef KERNELS_H
#define KERNELS_H
#include "SimulationList.h"
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <vector>
#include "Agent.h"

namespace Kernels
{
    episodeHistory simulateShared4(Agent &agent, int block, float *s, const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);


    episodeHistory simulateShared3(int block, float *s, const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

    void simulateShared2(int block, float *s, const int n, Simulation **sim, const float *allWeights, const float *allBiases, const float *startingParams, float *output);

    void mutate(int block, const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome, int *childSpecies, 
                           float *nextGenWeights, float *nextGenBiases, float * distances,  float * deltas, int * ancestors, float progThreshold, const int gen, const int shift);

    void mutate2(int block, const int n, const float randomMagnitude, const float *allWeights, const float *allBiases, float *simulationOutcome, 
                float *nextGenWeights, float *nextGenBiases, const int generation, const int shift, std::mt19937 & gen);
};

#endif