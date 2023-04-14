#include "BasicSimulation.cuh"

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

__device__ void BasicSimulation::eval(float **actions, float *gamestate)
{
    int tid = threadIdx.x;

    for (int bot = 0; bot < config_d.bpb; bot++)
    {
        // if (tid == 0)
        // {
        //     printf("layer shapes in derived %d\n", config_d.layerShapes[blockIdx.x % 3]);
        // }
        // if (tid < 4)
        // {
        //     //printf("block %d, tid %d : %f\n", blockIdx.x, tid, actions[bot][tid]);
        //     __syncthreads();
        //     actions[bot][tid + 1] += 1;
        // }
        tid++;
    }
}

__device__ int BasicSimulation::checkFinished(float *gamestate)
{
    return 0;
}

__host__ int BasicSimulation::getID()
{
    return 1;
}