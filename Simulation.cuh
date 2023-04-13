#include "cuda_runtime.h"
#include "math.h"
#include <math.h>

#include "device_launch_parameters.h"

#include <stdio.h>

//abstract class (interface) for what defines a valid simulation
#pragma once


//Not sure which one works better yet.
//typedef void (*SimulationLogic)(void*);
using SimulationLogic = void (*) (void*);


class Simulation {
public:
    virtual void configureKernelLaunchParameters() = 0;
    virtual void configureKernelFunctionParameters() = 0;
    virtual SimulationLogic getSimulationLogic() = 0;

};
