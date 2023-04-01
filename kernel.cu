//#ifndef debug
//#define DEBUG 1
//#endif


#include "cuda_runtime.h"
#include "math.h"
#include <math.h>

#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
void test();
void sharedTest();



//This is basically a #define. Needed for compile time understanding of certain variable definitions (particularly those that use numLayers for the size of an array)
constexpr auto numLayers = 3; //Number of layers;

constexpr auto bpt = 1; // Stands for "bots per thread" This allows for easy adjusting of the simulation if we wanna do something funky.


__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

//__device__ functions are called by individual GPU threads.
// This function computes the activations for a single bot.
// The layers array contains the weights and biases for each layer (except for input layer, which only has weights)
//__device__ void forward(float* layerShapes, float* layers, float* activations) {
//	
//	//This will hold the index offset for each layer
//	int l[numLayers];
//
//	//Calc the offset for each layer
//	l[0] = 0;
//	for (int i = 1; i < numLayers; i++) {
//		int numWeights = layerShapes[i - 1] * layerShapes[i];
//		int numBiases = layerShapes[i];
//		//we need the if statement here because the first layer (input layer) has only weights, not biases.
//		if (i == 1) {
//			l[i] = l[i - 1] + numWeights;
//		}
//		else {
//			l[i] = l[i - 1] + numWeights + numBiases;
//		}
//	}
//}

__host__ __device__ void printNet(float* activs, int* layerShapes) {
	printf("Activations:\n");
	int AO = 0; // "activs offset"

	for (int layer = 0; layer < numLayers; layer++) {
		printf("Layer %d, size = %d, AO = %d\n", layer, layerShapes[layer], AO);

		for (int i = 0; i < layerShapes[layer]; i++) {
			printf("%f, ", activs[AO + i]);
		}
		AO += layerShapes[layer];
		printf("\n");
	}
	printf("\n");
}



/**
 * Perform forward propagation of a dense neural network
 *
 * @param input     input data to the network, a float array of size input_size
 * @param weights   weight matrix of the network, a float array of size input_size * output_size
 * @param biases    bias vector of the network, a float array of size output_size
 * @param output    output of the network, a float array of size output_size
 * @param input_size    size of the input data
 * @param output_size   size of the output data
 */
__device__ void forward_propagation(const float* input, const float* weights, const float* biases, float* output, int input_size, int output_size)
{
	/*if (threadIdx.x == 0) {
		printf("Biases : ");
		for (int i = 0; i < output_size; i++) {
			printf("%f, ", biases[i]);
		}
		printf("\n");
	}*/
	// Initialize output to biases
	for (int i = 0; i < output_size; i++) {
		output[i] = biases[i];
	}

	/*if (threadIdx.x == 0) {
		printf("inside func Weights of layer:\n");
		
		int cc = 0;
		for (int k = 0; k < output_size * input_size; k++) {
			printf("%f, ", weights[k]);
			cc++;
			if (cc % output_size == 0)
				printf("\n");
		}
		printf("\n");
		printf("input size = %d, output size = %d\n", input_size, output_size);

	}*/
	
	// Compute dot product of input and weights
	for (int i = 0; i < input_size; i++) {
		for (int j = 0; j < output_size; j++) {
#ifdef DEBUG
			if (threadIdx.x == 0 && blockIdx.x == 0) {
				printf("adding %f * %f to %f, which is index %d. (i=%d)\n", input[i], weights[i * output_size + j], output[j], j, i);
				//printf("output: %p, input: %p\n", output[j], input[i]);
			}
#endif // DEBUG

			output[j] += input[i] * weights[i * output_size + j];
		}
	}

#ifdef DEBUG
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		printf("Activs : ");
		for (int i = 0; i < output_size; i++)
			printf("%f, ", output[i]);
		printf("\n\n");
	}
#endif // DEBUG

	


	// Apply activation function (sigmoid in this case)
	for (int i = 0; i < output_size; i++) {
		output[i] = 1 / (1 + expf(-output[i]));
	}
	
}


/*
* Note, I'll refer to the population as "bots"
* Brief description:
*   Using a dense neural network (NN)
*   There are N threads, corresponding to bpt * N bots. (these N's are lowercase in code)
*   networks memory layout in order:
		bpt * N * (b_ij + w_ij), where i=0...k, where k=numLayers is the number of layers
			and j=0...m, where m is the number of neurons in any layer.

			The effect of this memory layout is that each thread's NN data is contiguous in memory, rather than the data being
			interlaced.
*		Although the input layer doesn't have any biases, we will simply store them and have them set to 0 for code simplicity/cleanness
*
*   layerShapes is an array containing the size of each layer. There are numLayers layers.
*
*	The activations array is essentially scratch paper for all the intermediate steps while performing forward propagation.
*		NOTE: This scratch paper includes the inputs for each bot. This is to make the code for foward propagation cleaner.
*
*   Output will likely be an array of 2N length, with each entry corresponding to a bot's score. The host (CPU) will then
*       parse through the output and determine winners.
*
*/

__global__ void simulate(const int n, const float* networks, const int* layerShapes, const float* startingParams, float* activations, float* output) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x; //thread id
	

	//prevent OOB errors
	if (tid < n) {
		//calc the number of neurons per bot.
		int totalNeurons = 0;
		int totalWeights = 0;
		for (int i = 0; i < numLayers; i++) {
			if (i != numLayers - 1)
				totalWeights += layerShapes[i] * layerShapes[i + 1];
			totalNeurons += layerShapes[i];
		}

		//calculate this thread's memory offsets
		const float* threadNetworks = networks + (tid * (totalNeurons + totalWeights) * bpt); // * bpt here because each thread has bpt bots of data.
		float* threadActivations = activations + (tid * (totalNeurons) * bpt);

		//Seperate bot 1 and bot 2's nn data (weights and biases):
		const float* nns[bpt]; //abbreviation for "neural nets"

		//Seperate bot 1 and bot 2's "scratch paper" for NN calculations.
		float* activs[bpt]; //abbreviation for activations, but bot specific.

		//Populate the arrays created above
		for (int i = 0; i < bpt; i++) {
			nns[i] = threadNetworks + i * (totalNeurons + totalWeights);
			activs[i] = threadActivations + i * totalNeurons;
		}


		int maxIters = 500;
		bool finished = false;

		int iter = 0; //current timestep of simulation we're on

		//run the simulation loop.
		while (!finished) {
			//Determine inputs for the bot(s)
			for (int i = 0; i < bpt; i++) {
				for (int j = 0; j < layerShapes[0]; j++) {
					//This line is a placeholder for now.
					activs[i][j] = 0.5f;
				}				
			}


			//It's important to remember that activs and nns are essentially 2d arrays. That's why indexing them is tricky and weird.
			//Poll the NN for actions.
			for (int i = 0; i < bpt; i++) {
				int LO = 0; // "layer offset". used for indexing a bot's nns array.
				int AO = 0; // "activs offset"

				for (int layer = 0; layer < numLayers - 1; layer++) {
					int numWeights = layerShapes[layer] * layerShapes[layer + 1];
					int numBiases = layerShapes[layer];
					/*if (tid == 0) {
						printf("Weights of layer %d:\n", layer);
						/*for (int k = 0; k < numBiases; k++) {
							for (int l = 0; l < layerShapes[layer + 1]; l++) {
								printf("%f, ", (nns[i] + LO + numBiases)[k * layerShapes[layer + 1] + l]);
							}
							printf("\n");
						}
						int cc = 0;
						for (int k = 0; k < numWeights; k++) {
							printf("%f, ", (nns[i] + LO + numBiases)[k]);
							cc++;
							if (cc % layerShapes[layer + 1] == 0)
								printf("\n");
						}
						printf("\n");
					}
					*/
					//forward_propagation(float* input, float* weights, float* biases, float* output, int input_size, int output_size)
					forward_propagation(activs[i] + AO, nns[i] + LO + numBiases, nns[i] + numBiases + numWeights, activs[i] + AO + layerShapes[layer], layerShapes[layer], layerShapes[layer + 1]);

					AO += layerShapes[layer];
					LO += numBiases + numWeights;
				}
			}

			//update simulation/game state based on bot actions




			//do simulation/game logic



			//if(checkWinCondition(<something>)
			//	finished = true;

			iter++;
			if (iter >= maxIters) {
				finished = true;
			}
		}

		if (tid == 0 && blockIdx.x == 0) {
			printf("Activations:\n");
			int AO = 0; // "activs offset"
			for (int layer = 0; layer < numLayers; layer++) {
				printf("Layer %d, size = %d, AO = %d\n", layer, layerShapes[layer], AO);
				for (int i = 0; i < layerShapes[layer]; i++) {
					printf("%f, ", activs[0][AO + i]);
				}
				AO += layerShapes[layer];
				printf("\n");
			}
			printf("\n");
		}

	}
	return;
}

int main()
{

	test();


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (MHz): %d\n",
			prop.memoryClockRate / 1024);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
		printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
		printf("  minor-major: %d-%d\n", prop.minor, prop.major);
		printf("  Warp-size: %d\n", prop.warpSize);
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");
	}

	return 0;
}

void test() {
	
	int numThreads = 1024 * 16;
	int totalBots = numThreads * bpt;
	int numStartingParams = 1;
	int* layerShapes_h = new int[numLayers];
	float* startingParams_h = new float[numStartingParams];
	float* output_h = new float[totalBots];
	float* networks_h;

	int numConnections = 0;
	int numNeurons = 0;

	// Determine the network configuration
	layerShapes_h[0] = 6;
	layerShapes_h[1] = 6;
	layerShapes_h[2] = 4;

	// Calculate how many connections and neurons there are based on layerShapes_h so we can create the networks_h array.
	for (int i = 0; i < numLayers; i++) {
		if(i != numLayers - 1)
			numConnections += layerShapes_h[i] * layerShapes_h[i + 1]; //effectively numWeights
		numNeurons += layerShapes_h[i];	//effectively numBiases
	}
	int botNetSize = (numConnections + numNeurons); //how many indices a single bot uses in the networks_h array.
	int totalNetworksSize = botNetSize * totalBots;
	networks_h = new float[totalNetworksSize];
	printf("Total network size = %d KB\n", totalNetworksSize * sizeof(float) / (2 << 10));
	
	//initialize networks_h with random stuff for testing.

	for (int i = 0; i < totalBots; i++) {
		//printf("bot %d\n", i);
		int LO = 0;
		for (int j = 0; j < numLayers; j++) {		
			//printf("\tlayer %d\n", j);
			for (int k = 0; k < layerShapes_h[j]; k++) {
				//printf("\t\tNode %d: ", k);
				//set the biases
				if (j == 0) {
					//input layer biases are 0
					networks_h[i * botNetSize + LO + k] = 0;
				}
				else {
					//other layers get a bias = layer number.
					networks_h[i * botNetSize + LO + k] = j;
				}
				//printf("bias = %f, weights: ", networks_h[i * botNetSize + LO + k]);
				if (j != numLayers - 1) {
					for (int l = 0; l < layerShapes_h[j + 1]; l++) {
						//set the weights. all layers get a weight of layerNum+1
						networks_h[i * botNetSize + LO + layerShapes_h[j] + k * layerShapes_h[j + 1] + l] = j + 1;
						//printf("%f, ", networks_h[i * botNetSize + LO + layerShapes_h[j] + k * layerShapes_h[j + 1] + l]);
					}
				}
				
				//printf("\n");
				
			}
			if(j != numLayers - 1)
				LO += layerShapes_h[j] * layerShapes_h[j + 1] + layerShapes_h[j];
			//printf("\n");
		}
		//printf("\n");
	}

	//printNet(networks_h, layerShapes_h); //this doesn't work I think

	//Create pointers for device data
	
	int* layerShapes_d;

	float* startingParams_d;
	float* output_d;
	float* networks_d;
	float* activations_d;


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&layerShapes_d, numLayers * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaMalloc((void**)&startingParams_d, numStartingParams * sizeof(float));
	cudaMalloc((void**)&output_d, totalBots * sizeof(float));
	cudaMalloc((void**)&networks_d, totalNetworksSize * sizeof(float));
	cudaMalloc((void**)&activations_d, totalBots * numNeurons * sizeof(float));


	//copy data over to GPU
	cudaMemcpy(layerShapes_d, layerShapes_h, numLayers * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(startingParams_d, startingParams_h, numStartingParams * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(output_d, output_h, totalBots * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(networks_d, networks_h, totalNetworksSize * sizeof(float), cudaMemcpyHostToDevice);


	int tpb = 32; //threads per block
	int numBlocks = (numThreads + tpb - 1) / tpb;
	printf("Num blocks = %d\n", numBlocks);

	// Launch a kernel on the GPU with one thread for each element.
	simulate <<<numBlocks, tpb>>> (numThreads, networks_d, layerShapes_d, startingParams_d, activations_d, output_d);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulate!\n", cudaStatus);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output_h, output_d, numThreads * bpt * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//clean up memory.
Error:
	cudaFree(layerShapes_d);
	cudaFree(startingParams_d);
	cudaFree(output_d);
	cudaFree(networks_d);
	cudaFree(activations_d);

	
	delete[] layerShapes_h;
	delete[] startingParams_h;
	delete[] output_h;
	delete[] networks_h;

}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
