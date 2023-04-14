#include "Simulation.cuh"
#include "BasicSimulation.cuh"

extern __constant__ SimConfig config_d;

namespace Kernels{
    __global__ void game_kernel(int n, Simulation **sim)
    {

        // Placeholder for our activations/gamestate arrays.
        __shared__ float s1[64];
        __shared__ float s2[64];

        // global id
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid == 0)
        {
            printf("max iters = %d\n", config_d.maxIters);
        }
        // block-local id
        int tid = threadIdx.x;

        // Avoid OOB errors
        if (gid < n)
        {
            for (int i = 0; i < blockIdx.x + 2; i++)
            {
                (*sim)->eval(s1, s2);
                __syncthreads();
            }
        }
        return;
    }

    __global__ void createDerived(Simulation **sim,int id)
    {
        // From https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions
        // It is necessary to create object representing a function
        // directly in global memory of the GPU device for virtual
        // functions to work correctly, i.e. virtual function table
        // HAS to be on GPU as well.

        // NOTE: THIS SWITCH STATEMENT MUST BE UPDATED EACH TIME WE ADD A NEW SIMULATION!
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            switch (id)
            {
            case 1:
                (*sim) = new BasicSimulation();
                break;

            default:
                printf("Invalid derived class ID. Did you update the kernel switch statement?\n");
                break;
            }
        }

        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid == 0)
        {
            printf("total neurons = %d\n", config_d.totalNeurons);
        }

        return;
    };

    __global__ void delete_function(Simulation **sim)
    {
        delete *sim;
    };
};