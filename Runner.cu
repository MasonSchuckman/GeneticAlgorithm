#include "Simulation.cuh"
#include "BasicSimulation.cuh"

__global__ void game_kernel(int n, SimulationLogic game_logic, void* kernelArg) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n)
    printf("in kernel\n");
    //output[tid] = game_logic(input[tid]);
	(*game_logic)(kernelArg);
	return;
}

__device__ SimulationLogic logic_d;

void launchKernel(Simulation* derived, void* kernelArg) {
    derived->configureKernelLaunchParameters();
    derived->configureKernelFunctionParameters();
    SimulationLogic simLogic = derived->getSimulationLogic();
    //logic_d = simLogic;
    
    
    cudaSetDevice(0);

	SimulationLogic logic_h;
	cudaMemcpyFromSymbol(&logic_h, simLogic, sizeof(SimulationLogic));
    int n = 1;
	game_kernel<<<1, 1>>>(n, logic_h, kernelArg);
	cudaDeviceSynchronize();

	printf("done\n");
    // Code to launch the CUDA kernel with the configured parameters and function pointer
}

int main() {
    BasicSimulation sim;
    launchKernel(&sim, 0);
    return 0;
}