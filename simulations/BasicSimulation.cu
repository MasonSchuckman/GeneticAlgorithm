#include "BasicSimulation.cuh"

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

//Called at the beginning of the kernel. Used to do things like place the bots at their starting positions and such
__device__ void BasicSimulation::setupSimulation(const float * startingParams, float * gamestate){

}

//Called at the beginning of each sim iteration. 
__device__ void BasicSimulation::setActivations(float * gamestate, float ** activs, int iter){

}

__device__ void BasicSimulation::eval(float **actions, float *gamestate)
{
    int tid = threadIdx.x;
    if(tid == 0){
        printf("in basic eval\n");
    }
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