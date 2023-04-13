#include "cuda_runtime.h"
#include "math.h"
#include <math.h>

#include "device_launch_parameters.h"

#include <stdio.h>

struct DeviceFunctionStruct {
    void (*deviceFunctionPointer)(int);
};

__global__ void myKernel(DeviceFunctionStruct* deviceFunctionStruct) {
    if(threadIdx.x == 0)
        printf("in kernel\n");
    void (*deviceFunction)(int) = deviceFunctionStruct->deviceFunctionPointer;
    deviceFunction(42);
    
}

__device__ void myDeviceFunction(int arg) {
    printf("myDeviceFunction(%d)\n", arg);
}

int main() {
    DeviceFunctionStruct* deviceFunctionStruct;
    cudaMalloc((void**)&deviceFunctionStruct, sizeof(DeviceFunctionStruct));
    cudaMemcpyFromSymbol(&deviceFunctionStruct->deviceFunctionPointer, myDeviceFunction, sizeof(void*));

    myKernel<<<1, 1>>>(deviceFunctionStruct);
    cudaDeviceSynchronize();

    cudaFree(deviceFunctionStruct);
    return 0;
}
