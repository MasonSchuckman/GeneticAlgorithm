#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include "cuda_runtime.h"
#include "math.h"
#include <math.h>
#include "device_launch_parameters.h"
#include <stdio.h>

#include "Bot.h"
#include "biology/Taxonomy.h"
#include "biology/Specimen.h"
#include "SimulationList.cuh"
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
    Simulator(vector<Specimen*> bots, Simulation* derived, SimConfig &config, Taxonomy *history);

    ~Simulator();

    void simulate();

    

    void batchSimulate(int numSimulations);

    Bot * getBest();

    float mutateMagnitude = 1.0f; //starting magnitude
    float min_mutate_rate = .000001f; //ending magnitude
    float mutateDecayRate = 0.99f;
    float shiftEffectiveness = 1.0f;
    int loadData = 0;

private:

    void formatBotData(int *& layerShapes_h, float *&startingParams_h, 
        float *&output_h, float *&weights_h, float *&biases_h);

    void copyToGPU(int *& layerShapes_h, float *&startingParams_h, 
        float *&output_h, float *&weights_h, float *&biases_h);

    void copyFromGPU(float *&weights_h, float *&biases_h);


    void runSimulation(float * output_h, int *childSPecies_h);
    
    void readWeightsAndBiasesAll(float *&weights_h, float *&biases_h, int &TOTAL_BOTS, int &totalWeights, int &totalNeurons, int &numLayers, int * layerShapes);

    //Reads in saved weights and biases if it matches the current config
    void loadData_(float *weights_h, float *biases_h);

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
    int *parentSpecimen_d;

    float* nextGenWeights_d;
    float* nextGenBiases_d;


    vector<Specimen*> bots;
    Taxonomy *history;
    Simulation * derived;
    Simulation **sim_d;

    SimConfig config;

};

#endif