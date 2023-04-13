#pragma once

#include "Simulation.cuh"

class BasicSimulation : public Simulation {
public:
    void configureKernelLaunchParameters() override{}
    void configureKernelFunctionParameters() override{}
    
    SimulationLogic getSimulationLogic() override{
        m_kernelFunctionPtr = &logic;
        return m_kernelFunctionPtr;
    }

    __device__ void logic2(void*arg){
        if(threadIdx.x == 0)
        printf("basic\n");
    }

    //__device__ SimulationLogic func_d = logic2;

private:

    __device__ static void logic(void*arg){
        if(threadIdx.x == 0)
        printf("basic\n");
    }

    // Additional private variables and functions specific to DerivedClass
    SimulationLogic m_kernelFunctionPtr;
};
